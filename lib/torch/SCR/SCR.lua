-- SCR.lua - Cuda-aware solver for SCR-based regression
--
-- Copyright 2016, Ashton Fagg

local SCR = {}
------------------------------------
-- Dependancies
local t = require 'torch'
local c = require 'cutorch'
local m = require 'cephes'
local math = require 'math'

------------------------------------------
-- Utils

SCR.Util = {}
-- shitty hack to get trace working on GPU
function SCR.Util.trace(m)
   return torch.trace(m:float())
end

-- why does Torch not have a Kronecker product?
function SCR.Util.kron(A,B)
   local m, n = A:size(1), A:size(2)
   local p, q = B:size(1), B:size(2)
   local C = torch.CudaTensor(m*p,n*q)

   for i=1,m do
      for j=1,n do
         C[{{(i-1)*p+1,i*p},{(j-1)*q+1,j*q}}] = torch.mul(B, A[i][j])
      end
   end
   return C:cuda()
end

-- assumes that the rhs is a diagonal matrix
function SCR.Util.kronDiag(A,B)
   local m, n = A:size(1), A:size(2)
   local p, q = B:size(1), B:size(2)
   local C = torch.CudaTensor(m*p, n*q)
   C:zero()

   for i = 1,m do
      for j = 1, n do
         if i == j then
            C[{{(i-1)*p+1,i*p},{(j-1)*q+1,j*q}}] = torch.mul(B, A[i][j]) 
         end
      end
      
   end
   return C
end

-----------------------------------------------
-- This is the Kronecker factorization module
SCR.Util.KronFact = {}

function SCR.Util.KronFact.tilde(A, mb, nb)
   local m = A:size(1)
   local n = A:size(2)

   local mc = m / mb
   local nc = n / nb

   local T = torch.FloatTensor(mb*nb, mc*nc):zero()
   local x = torch.FloatTensor(1, mc*nc):zero()


   for ib = 1, mb do
      for jb = 1, nb do
         local rowSelect = torch.range((ib-1)*mc+1, ib*mc, 1):long()
         local colSelect = torch.range((jb-1)*nc+1, jb*nc, 1):long()
         local Asel = A:index(1, rowSelect):index(2, colSelect)
         x = torch.reshape(Asel, 1, mc*nc)
         T[{{(jb-1)*mb+ib}, {}}] = x
      end
   end
   return T
end

function SCR.Util.KronFact.factorize(C, mA, nA, mB, nB)
   C = C:float()
   local m = C:size(1)
   local n = C:size(2)
   local n1 = m/(mA*mB)
   local n3 = n/(nA*nB)

   local Ch = torch.FloatTensor(n1*mA*nA, n3*mB*nB)

   for i = 1, n1 do
      for j = 1, n3 do
         local ChrIdx = torch.range((i-1)*mA*nA+1, i*mA*nA, 1)
         local ChcIdx = torch.range((j-1)*mB*nB+1, j*mB*nB, 1)
         local CrIdx = torch.range((i-1)*mA*mB+1, i*mA*mB, 1):long()
         local CcIdx = torch.range((j-1)*nA*nB+1, j*nA*nB, 1):long()

         local CSel = C:index(1, CrIdx):index(2, CcIdx)
         local CSelTil = SCR.Util.KronFact.tilde(CSel, mA, nA)
         Ch[{{ChrIdx}, {ChcIdx}}] = CSelTil
      end
   end

   U,S,V = torch.svd(Ch)

   local s = torch.sqrt(S)
   local r = s:nonzero():size(1) -- this is the rank
   A = {}
   B = {}

   local sVW = torch.diag(s:index(1, torch.range(1,r,1):long()))
   local X = U:index(2, torch.range(1,r,1):long()) * sVW
   local Y = (V:index(2, torch.range(1,r,1):long()) * sVW):t()


   local A0 = torch.FloatTensor(mA, nA)
   local B0 = torch.FloatTensor(mB, nB)

   local pp = 1
   for i = 1, n1 do
      for j = 1, r do
         local rowSelect = torch.range((i-1)*mA*nA+1, i*mA*nA, 1):long();
         local colSelect = torch.LongTensor{j}
         
         local A0sel = X:index(1, rowSelect):index(2, colSelect)
         A0 = torch.reshape(A0sel, mA, nA):t() -- because row major
         A[pp] = A0
         pp = pp + 1
      end
   end

   pp = 1

   for i = 1, r do
      for j = 1, n3 do
         local rowSelect = torch.LongTensor{i};
         local colSelect = torch.range((j-1)*mB*nB+1, j*mB*nB):long()
         local B0sel = Y:index(1, rowSelect):index(2, colSelect)
         B0 = torch.reshape(B0sel, mB, nB)
         B[pp] = B0
         pp = pp + 1
      end
      
   end

   return A,B
end

function SCR.Util.KronFact.gradient(A, B)
   local nComponents = table.getn(A)
   local grad = torch.zeros(B[1]:size(1), B[1]:size(2)):float()

   for i = 1, nComponents do
      grad = grad + (B[i]*SCR.Util.trace(A[i]))
   end

   return grad:cuda()
end

function SCR.Util.KronFact.test()
   print('Testing KronFact...')
   local M = torch.randn(1568, 6272):float()
   local A, B = SCR.Util.KronFact.factorize(M, 49, 49, 32, 128);

   local Mvl = torch.zeros(49*32, 49*128):float()
   
   print('Forming estimate...')
   for i = 1, table.getn(A) do
      Mvl:add(SCR.Util.kron(A[i], B[i]):float())
   end

   local residual = torch.norm(Mvl-M) * torch.norm(Mvl-M)
   print('Error: ' .. math.sqrt(residual))
end


-- Does a deep copy of a table, useful for saving previous versions
-- of parameters and what not
function SCR.Util.tableDeepCopy(object)
   local lookup = {}
   local function _copy(object)
      if type(object) ~= "table" then
         return object
      end

      local newTable = {}
      lookup[object] = newTable
      for index, value in pairs(object) do
         newTable[_copy(index)] = _copy(value)
      end
      return setmetatable(newTable, getmetatable(object))
   end

   return _copy(object)
end


-----------------------------------------------------------------
-- Solvers

-------------------------------------------------------------------------
-- SolveSingleComponentGDScatterSim
-- Solves a single component SCR model using gradient descent, with no penalty
-- on any of the components. Instead of seeking convergence on the training set,
-- this method evaluates the performance of the current estimate on a validation
-- set.
-------------------------------------------------------------------------

function SCR.SolveSingleComponentGDScatterSim(trainStats, valStats, settings, composition, gpuID)
    print('Starting solver -> Single component Gradient Descent, scatter form, simultaneous xval.')
    c.setDevice(gpuID)

   -- Calculate the cost wrt to validation set
   local function objective(params, settings)
      local scale = 1 / (params.vYYt:size(1) * settings.nExamples)
      local cost = SCR.Util.trace(params.vYYt) + SCR.Util.trace(params.S_1 * params.vXXt * params.S_1:t())
      cost = cost + (SCR.Util.trace(params.S_1 * params.vXYt:t()) * -2)
      
      if m.isinf(cost) or m.isnan(cost) then
         return 1e99
      else
         return math.sqrt(scale*cost)
      end
   end

   local function gradS_1(params)
      local ret = (params.XXt * params.S_1:t()):t() * 2
      ret = ret + ((params.XYt:t()):t() * -2)
      return ret/t.norm(ret)
   end
   

   local params = {
      XXt = trainStats.XXt:cuda();
      YYt = trainStats.YYt:cuda();
      XYt = trainStats.XYt:cuda();
      S_1 = composition.S_1:cuda();
      vXXt = valStats.XXt:cuda();
      vYYt = valStats.YYt:cuda();
      vXYt = valStats.XYt:cuda();
   }

   local paramsPrev = SCR.Util.tableDeepCopy(params)
   local JBest, JCurr, JPrev = 1e99, {}, 1e99
   local paramsBest = SCR.Util.tableDeepCopy(params)
   local alpha = .01--1 / (settings.nExamples * params.YYt:size(1))
   
   for it = 1, settings.maxIts do
      local dJdS_1 = gradS_1(params)
      params.S_1 = params.S_1 + (dJdS_1 * -alpha)

      JCurr = objective(params, settings)
      print('Iteration: ' .. it .. ' Validation Cost: ' .. JCurr .. ' Best: ' .. JBest)
      
      -- Is this better than our best estimate thus far?
      if (JCurr < JBest) then
         JBest = JCurr
         paramsBest = SCR.Util.tableDeepCopy(params)
      end
      
      JPrev = JCurr
         
   end

   params = SCR.Util.tableDeepCopy(paramsBest)
      
   local SFinal = {
      S_1 = params.S_1:float();
   }
   print('JBest = ' .. JBest)
   
   return SFinal;
   
end


function SCR.SolveSparseCompGDScatterSim(trainStats, valStats, settings, composition, gpuID)
    print('Starting solver -> Sparse composition Gradient Descent, scatter form, simultaneous xval.')
    c.setDevice(gpuID)

   -- Calculate the cost wrt to validation set
    local function objective(params, settings)
      local scale = 1 / (params.vYYt:size(1) * settings.nExamples)
      local cost = SCR.Util.trace(params.vYYt)
      cost = cost + SCR.Util.trace(params.S_3*params.S_2* params.S_1 * params.vXXt * params.S_1:t() * params.S_2:t() * params.S_3:t())
      cost = cost + (SCR.Util.trace(params.S_3*params.S_2*params.S_1 * params.vXYt:t()) * -2)
      
      if m.isinf(cost) or m.isnan(cost) then
         return 1e99
      else
         return math.sqrt(scale*cost)
      end
   end

   local function gradS_1(params)
      local c1 = params.S_2:t() * params.S_3:t() * params.S_3 * params.S_2 * params.S_1 * params.XXt
      local c2 = params.S_2:t() * params.S_3:t() * params.XYt
      c1 = (c1*2) + (c2*-2)
      c1:cmul(params.M_1)
      return c1/t.norm(c1)
   end


   local function gradS_2(params)
      local c1 = (params.S_3:t() * params.S_3 * params.S_2 * params.S_1 * params.XXt * params.S_1:t()) * 2
      local c2 = (params.S_3:t() * params.XYt * params.S_1:t()) * -2
      c1 = c1 + c2
      c1:cmul(params.M_2)
      return c1/t.norm(c1)

   end

   local function gradS_3(params)
      local c1 = (params.S_3 * params.S_2 * params.S_1 * params.XXt * params.S_1:t() * params.S_2:t()) * 2
      local c2 = (params.XYt * params.S_1:t() * params.S_2:t()) * -2
      c1 = c1 + c2
      return c1/t.norm(c1)
   end
   
   

   local params = {
      XXt = trainStats.XXt:cuda();
      YYt = trainStats.YYt:cuda();
      XYt = trainStats.XYt:cuda();

      S_1 = composition.S_1:cuda();
      M_1 = composition.M_1:cuda();
            
      S_2 = composition.S_2:cuda();
      M_2 = composition.M_2:cuda();
      
      S_3 = composition.S_3:cuda();
      
      vXXt = valStats.XXt:cuda();
      vYYt = valStats.YYt:cuda();
      vXYt = valStats.XYt:cuda();
   }

   local paramsPrev = SCR.Util.tableDeepCopy(params)
   local JBest, JCurr, JPrev = 1e99, {}, 1e99
   local paramsBest = SCR.Util.tableDeepCopy(params)
   local alpha = .4--1 / (settings.nExamples * params.YYt:size(1))
   
   for it = 1, settings.maxIts do
      
      print('    Computing dJdS_1...')
      local dJdS_1 = gradS_1(params)
      params.S_1 = params.S_1 + (dJdS_1 * -alpha)
      
      print('    Computing dJdS_2...')
      local dJdS_2 = gradS_2(params)
      params.S_2 = params.S_2 + (dJdS_2 * -alpha)

      print('    Computing dJdS_3...')
      local dJdS_3 = gradS_3(params)
      params.S_3 = params.S_3 + (dJdS_3 * -alpha)

      JCurr = objective(params, settings)
      print('Iteration: ' .. it .. ' Validation Cost: ' .. JCurr .. ' Best: ' .. JBest)
      
      -- Is this better than our best estimate thus far?
      if (JCurr < JBest) then
         JBest = JCurr
         paramsBest = SCR.Util.tableDeepCopy(params)
      end
      
      JPrev = JCurr
         
   end

   params = SCR.Util.tableDeepCopy(paramsBest)
      
   local SFinal = {
      S_1 = params.S_1:float();
      S_2 = params.S_2:float();
      S_3 = params.S_3:float();
   }
   
   print('JBest = ' .. JBest)
   
   return SFinal;
   
end





function SCR.SolveFastGDScatterSim(trainStats, valStats, settings, composition, gpuID)
    print('Starting solver -> Fast composition Gradient Descent, scatter form, simultaneous xval.')
    c.setDevice(gpuID)

   -- Calculate the cost wrt to validation set
    local function objective(params, settings)
      --local S1f = SCR.Util.kron(params.I_1, params.S_1)
      local scale = 1 / (params.vYYt:size(1) * settings.nExamples)
      local cost = SCR.Util.trace(params.vYYt)
      cost = cost + SCR.Util.trace(params.S_3*params.S_2* params.S_1f * params.vXXt * params.S_1f:t() * params.S_2:t() * params.S_3:t())
      cost = cost + (SCR.Util.trace(params.S_3*params.S_2*params.S_1f* params.vXYt:t()) * -2)
      
      if m.isinf(cost) or m.isnan(cost) then
         return 1e99
      else
         return math.sqrt(scale*cost)
      end
   end

   local function gradS_1(params)
      local grad = torch.CudaTensor(params.S_1:size(1), params.S_1:size(2))
      local R = params.S_3 * params.S_2 * params.S_1f
      local B = params.S_3 * params.S_2
      M = ((params.XXt*R:t() - params.XYt:t()) * 2):t()
      M = B:t() * M
      -- Van Loan, Pitsianis kronecker approximation
      local A,B = SCR.Util.KronFact.factorize(M, params.I_1:size(1), params.I_1:size(2), params.S_1:size(1), params.S_1:size(2))
      grad = SCR.Util.KronFact.gradient(A,B)

      return grad/t.norm(grad)
   end


   local function gradS_2(params)
      --local S1f = SCR.Util.kron(params.I_1, params.S_1)
      local c1 = (params.S_3:t() * params.S_3 * params.S_2 * params.S_1f * params.XXt * params.S_1f:t()) * 2
      local c2 = (params.S_3:t() * params.XYt * params.S_1f:t()) * -2
      c1 = c1 + c2
      return c1/t.norm(c1)

   end

   local function gradS_3(params)
      --local S1f = SCR.Util.kron(params.I_1, params.S_1)
      local c1 = (params.S_3 * params.S_2 * params.S_1f * params.XXt * params.S_1f:t() * params.S_2:t()) * 2
      local c2 = (params.XYt * params.S_1f:t() * params.S_2:t()) * -2
      c1 = c1 + c2
      return c1/t.norm(c1)
   end
   
   

   local params = {
      XXt = trainStats.XXt:cuda();
      YYt = trainStats.YYt:cuda();
      XYt = trainStats.XYt:cuda();

      S_1 = composition.S_1:cuda();
      I_1 = composition.I_1:cuda();
      S_1f = torch.CudaTensor(composition.S_1:size(1)*composition.I_1:size(1), composition.S_1:size(2)*composition.I_1:size(2));
      
      S_2 = composition.M_2:cuda();
      M_2 = composition.M_2:cuda();
      
      S_3 = composition.S_3:cuda();
      
      vXXt = valStats.XXt:cuda();
      vYYt = valStats.YYt:cuda();
      vXYt = valStats.XYt:cuda();
   }

   local paramsPrev = SCR.Util.tableDeepCopy(params)
   local JBest, JCurr, JPrev = 1e99, {}, 1e99
   local paramsBest = SCR.Util.tableDeepCopy(params)
   local alpha = .3--1 / (settings.nExamples * params.YYt:size(1))
   params.S_1f = SCR.Util.kron(params.I_1, params.S_1);
   
   for it = 1, settings.maxIts do
      
      print('    Computing dJdS_1...')
      local dJdS_1 = gradS_1(params)
      params.S_1 = params.S_1 + (dJdS_1 * -alpha)
      params.S_1f = SCR.Util.kron(params.I_1, params.S_1); -- cache the full component

      print('    Computing dJdS_2...')
      local dJdS_2 = gradS_2(params)
      params.S_2 = params.S_2 + (dJdS_2 * -alpha)
      params.S_2:cmul(params.M_2) -- enforce desired sparsity pattern

      print('    Computing dJdS_3...')
      local dJdS_3 = gradS_3(params)
      params.S_3 = params.S_3 + (dJdS_3 * -alpha)

      JCurr = objective(params, settings)
      print('Iteration: ' .. it .. ' Validation Cost: ' .. JCurr .. ' Best: ' .. JBest)
      
      -- Is this better than our best estimate thus far?
      if (JCurr < JBest) then
         JBest = JCurr
         paramsBest = SCR.Util.tableDeepCopy(params)
      end
      
      JPrev = JCurr
         
   end

   params = SCR.Util.tableDeepCopy(paramsBest)
      
   local SFinal = {
      S_1 = params.S_1:float();
      S_2 = params.S_2:float();
      S_3 = params.S_3:float();
   }
   print('JBest = ' .. JBest)
   
   return SFinal;
   
end


---------------------------------------------------------------------------
-- End of package methods
return SCR

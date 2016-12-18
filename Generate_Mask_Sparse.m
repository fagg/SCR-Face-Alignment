clear all, close all;

%% S1 Mask

localMap = ones(64,128);
nPts = 49;

S1m = kron(eye(49), localMap);


%% S1 Init

localInit = eye(64,128);
S1 = kron(eye(49), localInit);

%% S2 Mask
S2m = zeros(1568, 3136);

hDim = size(S2m,2);
lDim = size(S2m,1);

hDim = hDim / 49;
lDim = lDim / 49;
rc = 1;
cc = 1;

leftEyeBrow = ones(5*lDim, 5*hDim);
rightEyeBrow = leftEyeBrow;

leftEye = ones(6*lDim, 6*hDim);
rightEye = leftEye;

nose = ones(9*lDim, 9*hDim);
mouth = ones(18*lDim, 18*hDim);

S2m(rc:rc+size(leftEyeBrow,1)-1, cc:cc+size(leftEyeBrow,2)-1) = leftEyeBrow;
rc = rc + size(leftEyeBrow,1);
cc = cc + size(leftEyeBrow,2);


S2m(rc:rc+size(rightEyeBrow,1)-1, cc:cc+size(rightEyeBrow,2)-1) = rightEyeBrow;
rc = rc + size(rightEyeBrow,1);
cc = cc + size(rightEyeBrow,2);


S2m(rc:rc+size(leftEye,1)-1, cc:cc+size(leftEye,2)-1) = leftEye;
rc = rc + size(leftEye,1);
cc = cc + size(leftEye,2);

S2m(rc:rc+size(rightEye,1)-1, cc:cc+size(rightEye,2)-1) = rightEye;
rc = rc + size(rightEye,1);
cc = cc + size(rightEye,2);

S2m(rc:rc+size(nose,1)-1, cc:cc+size(nose,2)-1) = nose;
rc = rc + size(nose,1);
cc = cc + size(nose,2);

S2m(rc:rc+size(mouth,1)-1, cc:cc+size(mouth,2)-1) = mouth;
rc = rc + size(mouth,1);
cc = cc + size(mouth,2);

%% S2 Init

S2 = zeros(1568, 3136);

hDim = size(S2,2);
lDim = size(S2,1);

hDim = hDim / 49;
lDim = lDim / 49;
rc = 1;
cc = 1;

leftEyeBrow = eye(5*lDim, 5*hDim);
rightEyeBrow = leftEyeBrow;

leftEye = eye(6*lDim, 6*hDim);
rightEye = leftEye;

nose = eye(9*lDim, 9*hDim);
mouth = eye(18*lDim, 18*hDim);

S2(rc:rc+size(leftEyeBrow,1)-1, cc:cc+size(leftEyeBrow,2)-1) = leftEyeBrow;
rc = rc + size(leftEyeBrow,1);
cc = cc + size(leftEyeBrow,2);


S2(rc:rc+size(rightEyeBrow,1)-1, cc:cc+size(rightEyeBrow,2)-1) = rightEyeBrow;
rc = rc + size(rightEyeBrow,1);
cc = cc + size(rightEyeBrow,2);


S2(rc:rc+size(leftEye,1)-1, cc:cc+size(leftEye,2)-1) = leftEye;
rc = rc + size(leftEye,1);
cc = cc + size(leftEye,2);

S2(rc:rc+size(rightEye,1)-1, cc:cc+size(rightEye,2)-1) = rightEye;
rc = rc + size(rightEye,1);
cc = cc + size(rightEye,2);

S2(rc:rc+size(nose,1)-1, cc:cc+size(nose,2)-1) = nose;
rc = rc + size(nose,1);
cc = cc + size(nose,2);

S2(rc:rc+size(mouth,1)-1, cc:cc+size(mouth,2)-1) = mouth;
rc = rc + size(mouth,1);
cc = cc + size(mouth,2);


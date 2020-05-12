function [features] = Extract_raw_GCC_PHAT(x,fs, wlen,hop, nfft)

chno=1:size(x,2); % Skippig the mic channels
nChan=length(chno);
nP= nChan*(nChan-1)/2; % Number of total combination
kp=zeros(nP,2);
cnt=1;

for i=1:nChan-1
    for j=i+1:nChan
        kp(cnt,:)=[chno(i),chno(j)];
        cnt=cnt+1;
    end
end

tau_grid = (-25:1:25)./fs;

feats1=zeros(nP,length(tau_grid));
for kk=1:nP
    kk1=kp(kk,1);
    kk2=kp(kk,2);
    X_spec = stft_v2(x(:,kk1), wlen, hop, nfft, fs);
    [Y_spec,f, ~] = stft_v2(x(:,kk2), wlen, hop, nfft, fs);
    spec = sum(squeeze(phat_spec(reshape([X_spec,Y_spec], [size([X_spec,Y_spec],1),1,size([X_spec,Y_spec],2)]), f.', tau_grid)),1);
    feats1(kk,:)=spec;
end
features=feats1(:)';
end

function [stft, f, t] = stft_v2(x, wlen, hop, nfft, fs)

% function: [stft, f, t] = stft(x, wlen, hop, nfft, fs)
% x - signal in the time domain
% wlen - length of the analysis Hamming window
% hop - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
% stft - STFT matrix (only unique points, time across columns, freq across rows)
% f - frequency vector, Hz
% t - time vector, s

% represent x as column-vector
x = x(:);

% length of the signal
xlen = length(x);

% form a periodic hamming window
win = hamming(wlen, 'periodic');

% stft matrix estimation and preallocation
rown = ceil((1+nfft)/2);            % calculate the total number of rows
coln = 1+fix((xlen-wlen)/hop);      % calculate the total number of columns
stft = zeros(rown, coln);           % form the stft matrix

% initialize the signal time segment index
indx = 0;

% perform STFT
for col = 1:coln
    % windowing
    xw = x(indx+1:indx+wlen).*win;

    % FFT
    X = fft(xw, nfft);

    % update the stft matrix
    stft(:, col) = X(1:rown);

    % update the index
    indx = indx + hop;
end

% calculate the time and frequency vectors
t = (wlen/2:hop:wlen/2+(coln-1)*hop)/fs;
f = (0:rown-1)*fs/nfft;

end

function spec = phat_spec(X, f, tau_grid)

% PHAT_SPEC Computes the GCC-PHAT spectrum as defined in
% C. Knapp, G. Carter, "The generalized cross-correlation method for
% estimation of time delay", IEEE Transactions on Acoustics, Speech and
% Signal Processing, 24(4):320–327, 1976.
%
% spec = phat_spec(X, f, tau_grid)
%
% Inputs:
% X: nbin x nfram x 2 matrix containing the STFT coefficients of the input
%     signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% tau_grid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nfram x ngrid array of angular spectrum values

[nbin,nfram] = size(X(:,:,1));
ngrid = length(tau_grid);
X1 = X(:,:,1);
X2 = X(:,:,2);

spec = zeros(nbin,nfram,ngrid);
P = X1.*conj(X2);
P = P./abs(P);
for ind = 1:ngrid
    EXP = repmat(exp(-2*1i*pi*tau_grid(ind)*f),1,nfram);
    spec(:,:,ind) = real(P.*EXP);
end

end
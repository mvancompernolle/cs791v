function str = abbr_num(x,d)
% abbreviate large numbers with suffix,
%   T (trillion), M (million), K (thousand)
%
% x is some large int
% d is the number of places to keep
%
% compress_num(23,456)
%   prints 23K
%
% compress_num(23,456,1)
%   prints 23.4K
%
%
% Ryan Rossi, Purdue University
% Copyright 2012
%

if nargin < 2,
    d = 1;
end

p = length(sprintf('%.f', abs(x)));
x = round(x);
str = num2str(x);


if p > 9,           %tril
    suffix = 'T';
    pdiff = p - 9;    
elseif p > 6,       %mil, p: 9,8,7
    suffix = 'M';
    pdiff = p - 6;
elseif p > 3,       %thous, p: 4,5,6
    suffix = 'K';
    pdiff = p - 3;
else
    suffix = '';
    pdiff = p;
end

if d == 0,
    str = [str(1:pdiff),suffix];
else
    if p >= 1 && p <= 3, 
        str = [str(1:pdiff),suffix];
    elseif strcmp(str(pdiff+1:pdiff+d),'0'), %case 379.0K
        str = [str(1:pdiff),suffix];
    else
        str = [str(1:pdiff),'.',str(pdiff+1:pdiff+d),suffix];
    end
end

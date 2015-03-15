function fields = get_fields(sgdata)
% takes as input a struct from cont2struct(gdata)
% removes str fields, vectors/matrices, etc..
%   - first ele used for retrieving relevant fields
%
% Ryan A. Rossi
% Copyright 2012, Purdue University
%

f = fieldnames(sgdata);

k = 1;
clear fields
for i=2:numel(f),
    if length(sgdata(1).(f{i})) == 1 || ischar(sgdata(1).(f{i}))
        d = strip_num(sgdata(1).(f{i}));
        if ~isnan(d),
            fields{k} = f{i};
            k=k+1;
        end
    end
end
fields

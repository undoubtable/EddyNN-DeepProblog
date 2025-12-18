% For each pixel, exactly one of {pos, neg, none} happens.
% You must provide these three probabilities per pixel and ensure they sum <= 1
% (ProbLog will normalize if needed, but最好自己归一化).

p_pos(Y,X)::state(Y,X,pos);
p_neg(Y,X)::state(Y,X,neg);
p_none(Y,X)::state(Y,X,none).

eddy_pos(Y,X) :- ocean(Y,X), state(Y,X,pos).
eddy_neg(Y,X) :- ocean(Y,X), state(Y,X,neg).

query(eddy_pos(Y,X)).
query(eddy_neg(Y,X)).

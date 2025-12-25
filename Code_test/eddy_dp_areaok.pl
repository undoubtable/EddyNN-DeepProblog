nn(seednet, [Img], Y, [0,1]) :: seed_present(Img, Y).

% AreaOK=1: output follows neural prediction
keep_label(Img, 1, 1) :- seed_present(Img, 1).
keep_label(Img, 1, 0) :- seed_present(Img, 0).

% AreaOK=0: always 0
keep_label(_Img, 0, 0).

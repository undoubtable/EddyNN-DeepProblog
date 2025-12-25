nn(seednet, [Img], Y, [0,1]) :: seed_present(Img, Y).

keep_label(Img, 1, 1) :- seed_present(Img, 1).
keep_label(Img, 1, 0) :- seed_present(Img, 0).
keep_label(_Img, 0, 0).

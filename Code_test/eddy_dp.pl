nn(seednet, [Img], Y, [0,1]) :: seed_present(Img, Y).

area_ok(A) :- A >= 15, A =< 400.

keep_label(Img, A, 1) :- area_ok(A), seed_present(Img, 1).
keep_label(Img, A, 0) :- area_ok(A), seed_present(Img, 0).
keep_label(_Img, A, 0) :- \+ area_ok(A).

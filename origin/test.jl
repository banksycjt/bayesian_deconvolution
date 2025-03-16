using Distributions 

dist = Normal(0, 1)
x = rand(dist, 10)
y = logpdf(dist())
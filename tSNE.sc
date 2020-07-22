/*
Ted Moore
ted@tedmooremusic.com
www.tedmooremusic.com
May 29, 2019

tSNE for use in SuperCollider (client side only)
ported from this javascript code:

https://github.com/karpathy/tsnejs/blob/master/tsne.js

*/
// create main global object
TSNE {
	var /*return_v,
	v_val,*/
	iter,
	perplexity,
	dim,
	epsilon,
	sizeOfDataSet,
	<y,
	gains,
	ystep;

	*new {
		arg perplexity = 30, dim = 2, epsilon = 10;
		^super.new.init(perplexity, dim, epsilon);
	}

	init {
		arg perplexity_ = 30, dim_ = 2, epsilon_ = 10;

		perplexity = perplexity_; // effective number of nearest neighbors
		dim = dim_; // by default 2-D tSNE
		epsilon = epsilon_; // learning rate

		iter = 0;
	}

	// utility function
	assert {
		arg condition, message;
		if(condition.not,{
			"Assertion failed".postln;
			Error(message).throw;
		});
	}

	// return random normal number
	randn {
		arg mu, std;
		^gauss(mu,std);
	}

	// utilitity that creates contiguous vector of zeros of size n
	zeros {
		arg n;
		^DoubleArray.newClear(n); // typed arrays are faster
	}

	// utility that returns 2d array filled with random numbers
	// or with value s, if provided
	randn2d {
		arg n,d,s;
		var x = List.new;
		n.do({
			arg i;
			var xhere = List.new;
			d.do({
				arg j;
				if(s.notNil,{
					xhere.add(s);
				},{
					xhere.add(this.randn(0.0, 1e-4));
				})
			});
			x.add(xhere);
		});
		^x;
	}

	// compute L2 distance between two vectors
	l2 {
		arg x1, x2;
		var len = x1.size;
		var d = 0;
		len.do({
			arg i;
			var x1i = x1[i];
			var x2i = x2[i];
			d = d + ((x1i-x2i)*(x1i-x2i));
		})
		^d;
	}

	// compute pairwise distance in all vectors in X
	xtod {
		arg x;
		var n, dist;
		n = x.size;
		dist = this.zeros(n * n); // allocate contiguous array
		n.do({
			arg i;
			var j;
			j = i + 1;
			while({j < n},{
				var d;
				//"xtod j: %".format(j).postln;
				d = this.l2(x[i], x[j]);
				//"xtod n: %".format(n).postln;
				//"xtod d: %".format(d).postln;
				dist[(i*n)+j] = d;
				dist[(j*n)+i] = d;
				j = j + 1;
			});
			"xtod i: % of %".format(i+1,n).postln;

		});
		^dist;
	}

	// compute (p_{i|j} + p_{j|i})/(2n)
	d2p {
		arg d, perplexity, tol;
		var nf, n, htarget, p, prow, pout, n2;
		nf = d.size.sqrt; // this better be an integer
		n = nf.floor;
		this.assert(n == nf, "D should have square number of elements.");
		htarget = perplexity.log; // target entropy of distribution
		p = this.zeros(n * n); // temporary probability matrix

		prow = this.zeros(n); // a temporary storage compartment

		n.do({
			arg i;
			var betamin = -inf;
			var betamax = inf;
			var beta = 1; // initial value of precision
			var done = false;
			var maxtries = 50;

			// perform binary search to find a suitable precision beta
			// so that the entropy of the distribution is appropriate
			var num = 0;

			while({done.not},{
				var psum, hhere;
				//debugger;

				// compute entropy and kernel row with beta precision
				psum = 0.0;
				n.do({
					arg j;
					var pj = 2.71828.pow(-1 * d[(i*n)+j] * beta);
					if(i==j,{pj = 0}); // we dont care about diagonals
					prow[j] = pj;
					psum = psum + pj;
				});
				// normalize p and compute entropy
				hhere = 0.0;
				n.do({
					arg j;
					var pj;
					if(psum == 0,{
						pj = 0;
					},{
						pj = prow[j] / psum;
					});
					prow[j] = pj;
					if(pj > 1e-7,{
						hhere = hhere - (pj * pj.log);
					});
				});

				// adjust beta based on result
				if(hhere > htarget,{
					// entropy was too high (distribution too diffuse)
					// so we need to increase the precision for more peaky distribution
					betamin = beta; // move up the bounds
					if(betamax == inf,{
						beta = beta * 2;
					},{
						beta = (beta + betamax) / 2;
					});
				},{
					// converse case. make distrubtion less peaky
					betamax = beta;
					if(betamin == (-inf),{
						beta = beta / 2;
					},{
						beta = (beta + betamin) / 2;
					});
				});

				// stopping conditions: too many tries or got a good precision
				num = num + 1;
				if((hhere - htarget).abs < tol,{done = true});
				if(num >= maxtries,{done = true});
			});

			// console.log('data point ' + i + ' gets precision ' + beta + ' after ' + num + ' binary search steps.');
			// copy over the final prow to P at row i
			n.do({
				arg j;
				p[(i*n)+j] = prow[j];
			});

			"d2p i: % of %".format(i+1,n).postln;

		}); // end loop over examples i

		// symmetrize P and normalize it to sum to 1 over all ij
		pout = this.zeros(n * n);
		n2 = n*2;
		n.do({
			arg i;
			n.do({
				arg j;
				pout[(i*n)+j] = max((p[(i*n)+j] + p[(j*n)+i])/n2, 1e-100);
			});
		});

		^pout;
	}

	// this function takes a set of high-dimensional points
	// and creates matrix P from them using gaussian kernel
	initDataRaw {
		arg x;
		var n = x.size, d = x[0].size, dists;
		//n = x.size;
		//d = x[0].size;
		this.assert(n > 0, " X is empty? You must have some data!");
		this.assert(d > 0, " X[0] is empty? Where is the data?");
		dists = this.xtod(x); // convert X to distances using gaussian kernel
		perplexity = this.d2p(dists, perplexity, 1e-4); // attach to object
		sizeOfDataSet = n; // back up the size of the dataset
		this.initSolution; // refresh this
	}

	// this function takes a given distance matrix and creates
	// matrix P from them.
	// D is assumed to be provided as a list of lists, and should be symmetric
	initDataDist {
		arg d;
		var n, dists;
		n = d.size;
		assert(n > 0, " X is empty? You must have some data!");
		// convert D to a (fast) typed array version
		dists = this.zeros(n * n); // allocate contiguous array
		n.do({
			arg i;
			for(i+1,n-1,{
				arg j;
				var dp = d[i][j];
				dists[(i*n)+j] = dp;
				dists[(j*n)+i] = dp;
			})
		});
		perplexity = this.d2p(dists, perplexity, 1e-4);
		sizeOfDataSet = n;
		this.initSolution; // refresh this
	}

	// (re)initializes the solution to random
	initSolution {
		// generate random solution to t-SNE
		y = this.randn2d(sizeOfDataSet, dim); // the solution
		gains = this.randn2d(sizeOfDataSet, dim, 1.0); // step gains to accelerate progress in unchanging directions
		ystep = this.randn2d(sizeOfDataSet, dim, 0.0); // momentum accumulator
		iter = 0;
	}

	// return pointer to current solution
	getSolution {
		^y;
	}

	// perform a single step of optimization to improve the embedding
	step {
		var n, cg, cost, grad, ymean;
		"iter: %".format(iter).postln;
		iter = iter + 1;
		n = sizeOfDataSet;

		cg = this.costGrad(y); // evaluate gradient
		cost = cg.cost;
		grad = cg.grad;

		// perform gradient step
		ymean = this.zeros(dim);
		n.do({
			arg i;
			dim.do({
				arg d;
				var gid, sid, gainid, newgain, momval, newsid;
				gid = grad[i][d];
				sid = ystep[i][d];
				gainid = gains[i][d];

				// compute gain update
				if(gid.sign == sid.sign,{
					newgain = gainid * 0.8
				},{
					newgain = gainid + 0.2
				});
				if(newgain < 0.01,{newgain = 0.01}); // clamp
				gains[i][d] = newgain; // store for next turn

				// compute momentum step direction
				if(iter < 250,{momval = 0.5},{momval = 0.8});
				newsid = (momval * sid) - (epsilon * newgain * grad[i][d]);
				ystep[i][d] = newsid; // remember the step we took

				// step!
				y[i][d] = y[i][d] + newsid;

				ymean[d] = ymean[d] + y[i][d]; // accumulate mean so that we can center later
			})
		});

		// reproject Y to be zero mean
		n.do({
			arg i;
			dim.do({
				arg d;
				y[i][d] = y[i][d] - (ymean[d]/n);
			});
		});

		^cost;
	}

	// for debugging: gradient check
	debugGrad {
		var n, cg, cost, grad, e;
		n = sizeOfDataSet;

		cg = this.costGrad(y); // evaluate gradient
		cost = cg.cost;
		grad = cg.grad;

		e = 1e-5;
		n.do({
			arg i;
			dim.do({
				arg d;
				var yold, cg0, cg1, analytic,numerical;
				yold = y[i][d];

				y[i][d] = yold + e;
				cg0 = this.costGrad(y);

				y[i][d] = yold - e;
				cg1 = this.costGrad(y);

				analytic = grad[i][d];
				numerical = (cg0.cost - cg1.cost) / ( 2 * e );
				(i + "," + d + ": gradcheck analytic: " + analytic + " vs. numerical: " + numerical).postln;

				y[i][d] = yold;
			})
		})
	}

	// return cost and gradient, given an arrangement
	costGrad {
		arg y;
		var n, p, pmul, qu, qsum, nn, q, cost, grad;
		n = sizeOfDataSet;
		p = perplexity;

		if(iter < 100,{pmul = 4},{pmul = 1}); // trick that helps with local optima

		// compute current Q distribution, unnormalized first
		qu = this.zeros(n * n);
		qsum = 0.0;
		n.do({
			arg i;
			var j = i + 1;
			while({j < n},{
				var dsum = 0.0, qu1;
				dim.do({
					arg d;
					var dhere = y[i][d] - y[j][d];
					dsum = dsum + (dhere * dhere);
				});
				qu1 = 1.0 / (1.0 + dsum); // Student t-distribution
				qu[(i*n)+j] = qu1;
				qu[(j*n)+i] = qu1;
				qsum = qsum + (2 * qu1);
				j = j + 1;
			})
		});
		// normalize Q distribution to sum to 1
		nn = n*n;
		q = this.zeros(nn);
		nn.do({
			arg q1;
			q[q1] = max(qu[q1] / qsum, 1e-100);
		});

		cost = 0.0;
		grad = List.new;
		n.do({
			arg i;
			var gsum = Array.fill(dim,{0}); // init grad for point i
			dim.do({arg d; gsum[d] = 0.0; });
			n.do({
				arg j;
				var premult;
				cost = cost + (-1 * p[(i*n)+j] * q[(i*n)+j].log); // accumulate cost (the non-constant portion at least...)
				premult = 4 * (pmul * p[(i*n)+j] - q[(i*n)+j]) * qu[(i*n)+j];
				dim.do({
					arg d;
					gsum[d] = gsum[d] + (premult * (y[i][d] - y[j][d]));
				});
			});
			grad.add(gsum);
		});

		^(cost: cost, grad: grad.deepCopy);
	}
}
#include"layer.h"
#include"matrix.h"
#include<iostream>
#include<vector>
#include<string>
#include <iomanip>
#include<random>
using namespace std;

layer::layer(int N, int M, vector<double> (*F)(const vector<double>&), vector<double> (*Df)(const vector<double>&))
{
	n = N;
	m = M;
	w.resize(m);
	for (int i = 0; i < m; i++)
	{
		w[i].resize(n);
	}
	b.resize(m);
	f = F;
	df = Df;
}

bool layer::const_init(double w_const, double b_const)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			w[i][j] = w_const;
		}
		b[i] = b_const;
	}
	return true;
}

bool layer::gaussian_init(double mean, double var)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			w[i][j] = gaussrand() * var + mean;
		}
		b[i] = gaussrand() * var + mean;
	}
	return true;
}

bool layer::positive_unitball_init()
{
	for (int i = 0; i < m; i++)
	{
		double sum = 0;
		for (int j = 0; j < n; j++)
		{
			w[i][j] = gaussrand();
			sum += w[i][j];
		}
		w[i] = w[i] * (1.0 / sum);
		b[i] = gaussrand();
	}
	return true;
}

bool layer::uniform_init(double mi, double mx)
{
	default_random_engine random(time(NULL));
	uniform_real_distribution<double> dist(mi, mx);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			w[i][j] = dist(random);
		}
		b[i] = dist(random);
	}
	return true;
}

bool layer::xavier_init()
{
	default_random_engine random(time(NULL));
	uniform_real_distribution<double> dist(-sqrt(6.0 / (m + n)), sqrt(6.0 / (m + n)));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			w[i][j] = dist(random);
		}
		b[i] = dist(random);
	}
	return true;
}

vector<vector<double> > layer::get_w()
{
	return w;
}

vector<double> layer::get_b()
{
	return b;
}

bool layer::set_w(const vector<vector<double> >& W)
{
	w.assign(W.begin(), W.end());
	return true;
}

bool layer::set_b(const vector<double>& B)
{
	b.assign(B.begin(), B.end());
	return true;
}

vector<double> layer::y(const vector<double>& x)
{
	vector<double> output = flat(w * T(x));
	output = output + b;
	output = f(output);
	return output;
}

vector<double> layer::dx(const vector<double>& x, const vector<double>& dy)
{
	vector<double> dX;
	vector<vector<double> > dY;
	dY.push_back(df(y(x)) * dy);
	dX = flat(dY * w);
	return dX;
}

vector<vector<double> > layer::dw(const vector<double>& x, const vector<double>& dy)
{
	vector<double> d(m,1.0);
	d = df(y(x));
	vector<vector<double> > X;
	X.push_back(x);
	vector<vector<double> > dW = T(d * dy) * X;
	return dW;
}

vector<double> layer::db(const vector<double>& x, const vector<double>& dy)
{
	vector<double> d(m, 1.0);
	d = df(y(x));
	vector<double > dB = d * dy;
	return dB;
}


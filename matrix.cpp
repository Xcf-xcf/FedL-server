#include"matrix.h"
#include<iostream>
#include<cmath>
#include<algorithm>
#include <time.h>
using namespace std;

vector<vector<double> > eye(const int& n, const int& m)
{
	vector<vector<double> > x(n,vector<double>(m));
	int mi = min(n, m);
	for (int i = 0; i < mi; i++)
	{
		x[i][i] = 1;
	}
	return x;
}

vector<vector<double> > diag(const vector<double>& a)
{
	int siz = a.size();
	vector<vector<double> > x(siz, vector<double>(siz));
	for (int i = 0; i < siz; i++)
	{
		x[i][i] = a[i];
	}
	return x;
}

vector<vector<double> > T(const vector<vector<double> >& a)
{
	int n = a[0].size(), m = a.size();
	vector<vector<double> > x(n,vector<double>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] = a[j][i];
		}
	}
	return x;
}

vector<vector<double> > T(const vector<double>& a)
{
	vector<vector<double> > x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i].push_back(a[i]);
	}
	return x;
}

vector<double> flat(const vector<vector<double> >& a)
{
	vector<double> x;
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < a[i].size(); j++)
		{
			x.push_back(a[i][j]);
		}
	}
	return x;
}

vector<vector<double> > repeat(const vector<double>& a, const int& n, const int& axis)
{
	vector<vector<double> > x(n, vector<double>(a.begin(), a.end()));
	if (axis == 1)
		x = T(x);
	return x;
}


double det(vector<vector<double> > a)
{
	int n = a.size();
	for (int i = 0; i < n; i++)
	{
		if (a[i][i] == 0)
		{
			for (int j = i + 1; j < n; j++)
			{
				if (a[j][i] != 0)
				{
					swap(a[i], a[j]);
					break;
				}
			}
			if (a[i][i] == 0)
				return 0;
		}
		for (int k = i + 1; k < n; k++)
		{
			double t = a[k][i] / a[i][i];
			for (int j = i; j < n; j++)
			{
				a[k][j] -= t * a[i][j];
			}
		}
	}
	double x = 1;
	for (int i = 0; i < n; i++)
	{
		x *= a[i][i];
	}
	return x;
}

vector<vector<double> > operator*(const vector<vector<double> >& a, const vector<vector<double> >& b)
{
	int n = a.size(), m = b[0].size(), d=a[0].size();
	vector<vector<double> > x(n, vector<double>(m,0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			for (int k = 0; k < d; k++)
			{
				x[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return x;
}

vector<vector<double> > operator*(const vector<vector<double> >& a, const double& b)
{
	vector<vector<double> > x = a;
	int n = x.size(), m = x[0].size();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] *= b;
		}
	}
	return x;
}

vector<vector<double> > operator*(const double& b, const vector<vector<double> >& a)
{
	vector<vector<double> > x = a;
	int n = x.size(), m = x[0].size();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] *= b;
		}
	}
	return x;
}

vector<double> operator*(const vector<double>& a, const double& b)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] * b;
	}
	return x;
}

vector<double> operator*(const double& b, const vector<double>& a)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] * b;
	}
	return x;
}
vector<double> operator*(const vector<double>& a, const vector<double>& b)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] * b[i];
	}
	return x;
}

vector<vector<double> > operator/(const vector<vector<double> >& a, const vector<vector<double> >& b)
{
	int n = a.size(), m = a[0].size();
	vector<vector<double> > x(n, vector<double>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] = a[i][j] / b[i][j];
		}
	}
	return x;
}

vector<vector<double> > operator+(const vector<vector<double> >& a, const vector<vector<double> >& b)
{
	int n = a.size(), m = a[0].size();
	vector<vector<double> > x(n, vector<double>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] = a[i][j] + b[i][j];
		}
	}
	return x;
}

vector<vector<double> > operator+(const vector<vector<double> >& a, const double& b)
{
	int n = a.size(), m = a[0].size();
	vector<vector<double> > x(n, vector<double>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] = a[i][j] + b;
		}
	}
	return x;
}

vector<vector<double> > operator+(const double& b, const vector<vector<double> >& a)
{
	int n = a.size(), m = a[0].size();
	vector<vector<double> > x(n, vector<double>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] = a[i][j] + b;
		}
	}
	return x;
}

vector<double> operator+(const vector<double>& a, const vector<double>& b)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] + b[i];
	}
	return x;
}

vector<double> operator+(const vector<double>& a, const double& b)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] + b;
	}
	return x;
}

vector<double> operator+(const double& b, const vector<double>& a)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] + b;
	}
	return x;
}

vector<vector<double> > operator-(const vector<vector<double> >& a, const vector<vector<double> >& b)
{
	int n = a.size(), m = a[0].size();
	vector<vector<double> > x(n, vector<double>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			x[i][j] = a[i][j] - b[i][j];
		}
	}
	return x;
}

vector<double> operator-(const vector<double>& a, const vector<double>& b)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] - b[i];
	}
	return x;
}

vector<double> operator-(const vector<double>& a, const double& b)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = a[i] - b;
	}
	return x;
}

vector<double> operator-(const double& b, const vector<double>& a)
{
	vector<double> x(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		x[i] = b - a[i];
	}
	return x;
}

vector<bool> operator&&(const vector<bool>& a, const vector<bool>& b)
{
	int n = a.size();
	vector<bool> x(n);
	for (int i = 0; i < n; i++)
	{
		x[i] = (a[i] && b[i]);
	}
	return x;
}

int sum(const vector<bool>& a)
{
	int n = a.size();
	int x = 0;
	for (int i = 0; i < n; i++)
	{
		if (a[i])
			x++;
	}
	return x;
}

vector<vector<double> > row(const vector<vector<double> >& a, const vector<bool>& b)
{
	vector<vector<double> > x;
	int n = a.size();
	for (int i = 0; i < n; i++)
	{
		if (b[i])
			x.push_back(a[i]);
	}
	return x;
}

vector<vector<double> > col(const vector<vector<double> >& a, const vector<bool>& b)
{
	vector<vector<double> > A = T(a);
	vector<vector<double> > x = T(row(A, b));
	return x;
}

vector<int> argmax(const vector<vector<double> >& a, const int& axis)
{
	vector<int> index;
	int n = a.size(), m = a[0].size();
	if (axis)
	{
		for (int i = 0; i < n; i++)
		{
			double mx=a[i][0];
			index.push_back(0);
			for (int j = 1; j < m; j++)
			{
				if (a[i][j] > mx)
				{
					mx = a[i][j];
					index[i] = j;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < m; i++)
		{
			double mx = a[0][i];
			index.push_back(0);
			for (int j = 1; j < n; j++)
			{
				if (a[j][i] > mx)
				{
					mx = a[j][i];
					index[i] = j;
				}
			}
		}
	}
	return index;
}

int argmax(const vector<double>& a)
{
	int index = 0;
	double mx = a[0];
	for (int i = 0; i < a.size(); i++)
	{
		if (a[i] > mx)
		{
			mx - a[i];
			index = i;
		}
	}
	return index;
}

vector<vector<double> > inv(const vector<vector<double> >& a)
{
	int n = a.size();
	vector<vector<double> > D(n, vector<double>(2 * n, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < 2 * n; j++)
		{
			if (j < n)
			{
				D[i][j] = a[i][j];
			}
			else
			{
				if ((j - n) == i)
				{
					D[i][j] = 1;
				}
			}
		}
	}
	for (int i = 0; i < n; i++)
	{
		if (D[i][i] == 0)
		{
			for (int j = i + 1; j < n; j++)
			{
				if (D[j][i] != 0)
				{
					swap(D[i], D[j]);
				}
			}
			if (D[i][i] == 0)
			{
				throw - 1;
			}
		}
		double t = D[i][i];
		for (int j = i; j < 2*n; j++)
		{
			D[i][j] /= t;
		}
		for (int k = 0; k < n; k++)
		{
			if (k == i)continue;
			t = -D[k][i];
			for (int j = i; j < 2 * n; j++)
			{
				D[k][j] += t * D[i][j];
			}
		}
	}
	vector<vector<double> > x(n, vector<double>(n, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			x[i][j] = D[i][j + n];
		}
	}
	return x;
}

vector<double> mean(const vector<vector<double> >& a, int axis)
{
	if (axis == 0)
	{
		int m = a[0].size(), n=a.size();
		vector<double> x(m);
		for (int j = 0; j < m; j++)
		{
			for (int i = 0; i < n; i++)
			{
				x[j] += a[i][j];
			}
			x[j] /= n;
		}
		return x;
	}
	else
	{
		int m = a[0].size(), n = a.size();
		vector<double> x(n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				x[i] += a[i][j];
			}
			x[i] /= m;
		}
		return x;
	}
}

vector<vector<double> > cov(const vector<vector<double> >& a, bool colvar)
{
	if (colvar)
	{
		vector<double> b = mean(a);
		vector<vector<double> > avg = repeat(b, a.size());
		vector<vector<double> > x = a + avg * (-1);
		vector<vector<double> > var = T(x) * x;
		return var;
	}
	else
	{
		vector<double> b = mean(a, 1);
		vector<vector<double> > avg = repeat(b, a.size(), 1);
		vector<vector<double> > x = a + avg * (-1);
		vector<vector<double> > var = x * T(x);
		return var;
	}
}

vector<double> Y(const vector<double>& x)
{
	vector<double> y(x);
	return y;
}

vector<double> Dy(const vector<double>& x)
{
	vector<double> dy(x.size(), 1.0);
	return dy;
}

vector<double> sigmoid(const vector<double>& x)
{
	vector<double> y;
	for (int i = 0; i < x.size(); i++)
	{
		double cnt = 1.0 / (1 + exp(-x[i]));
		y.push_back(cnt);
	}
	return y;
}

vector<double> dsigmoid(const vector<double>& x)
{
	vector<double> dy = sigmoid(x) * (1 - sigmoid(x));
	return dy;
}

vector<double> softmax(const vector<double>& x)
{
	double sum = 0;
	vector<double> y;
	for (int i = 0; i < x.size(); i++)
	{
		y.push_back(exp(x[i]));
		sum += exp(x[i]);
	}
	y = (1.0 / sum) * y;
	return y;
}

vector<double> dsoftmax(const vector<double>& x)
{
	vector<double> dy(x.size(), 1.0);
	return dy;
}

vector<double> relu(const vector<double>& x)
{
	vector<double> y;
	for (int i = 0; i < x.size(); i++)
	{
		y.push_back(max(x[i], 0.0));
	}
	return y;
}

vector<double> drelu(const vector<double>& x)
{
	vector<double> y;
	for (int i = 0; i < x.size(); i++)
	{
		if (x[i] > 0)
			y.push_back(1.0);
		else
			y.push_back(0.0);
	}
	return y;
}

double MSE(const vector<double>& y, const vector<double>& Y)
{
	double loss=0;
	vector<double> d = (y - Y) * (y - Y);
	for (int i = 0; i < d.size(); i++)
		loss += d[i];
	loss *= 1.0/2.0;
	return loss;
}

double MSE(const vector<vector<double> >& y, const vector<vector<double> >& Y)
{
	double loss = 0;
	int n = y.size();
	for (int i = 0; i < n; i++)
	{
		loss += MSE(y[i], Y[i]);
	}
	loss /= 2*n;
	return loss;
}

vector<double> dMSE(const vector<double>& y, const vector<double>& Y)
{
	vector<double> dy = (y - Y);
	return dy;
}

double CrossEntropy(const vector<double>& y, const vector<double>& Y)
{
	double loss = 0;
	for (int i = 0; i < y.size(); i++)
	{
		loss -= Y[i] * log(y[i]);
	}
	return loss;
}

double CrossEntropy(const vector<vector<double> >& y, const vector<vector<double> >& Y)
{
	double loss = 0;
	int n = y.size();
	for (int i = 0; i < n; i++)
	{
		loss += CrossEntropy(y[i], Y[i]);
	}
	loss /= n;
	return loss;
}

vector<double> dCrossEntropy(const vector<double>& y, const vector<double>& Y)
{
	vector<double> dy = y - Y;
	return dy;
}

double gaussrand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;
	srand(time(NULL));
	if (phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}
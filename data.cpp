#include"data.h"
#include"matrix.h"
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<algorithm>
#include<ctime>
#include<cstdio>
using namespace std;

bool read_in(vector<vector<double> >& data, vector<int>& labels, string dataPath, string labelsPath)
{
	ifstream dataFile,labelsFile;
	dataFile.open(dataPath.data());
	labelsFile.open(labelsPath.data());
	string s="";
	while (getline(dataFile, s))
	{
		stringstream ss(s);
		vector<double> x;
		double buf;
		while (ss >> buf)
		{
			x.push_back(buf);
		}
		data.push_back(x);
	}
	int x;
	while (labelsFile >> x)
	{
		labels.push_back(x);
	}
	return true;
}

bool read_in(vector<vector<double> >& data, vector<vector<double> >& y, string dataPath, string yPath)
{
	ifstream dataFile, yFile;
	dataFile.open(dataPath.data());
	yFile.open(yPath.data());
	string s = "";
	while (getline(dataFile, s))
	{
		stringstream ss(s);
		vector<double> x;
		double buf;
		while (ss >> buf)
		{
			x.push_back(buf);
		}
		data.push_back(x);
	}
	while (getline(yFile, s))
	{
		stringstream ss(s);
		vector<double> x;
		double buf;
		while (ss >> buf)
		{
			x.push_back(buf);
		}
		y.push_back(x);
	}
	return true;
}

bool split_data(vector<vector<vector<double> > >& data_sect, vector<vector<int> >& labels_sect, const vector<vector<double> >& data, const vector<int>& labels, const vector<double>& ratio)
{
	int n = data.size();
	vector<int> index;
	for (int i = 0; i < n; i ++ )
	{
		index.push_back(i);
	}
	srand(unsigned(time(0)));
	random_shuffle(index.begin(), index.end());
	double sum = ratio[0];
	int num = sum * n;
	vector<vector<double> > d;
	vector<int> l;
	for (int i = 0, j = 0; i < n; i++)
	{
		if(i >= num)
		{
			data_sect.push_back(d);
			labels_sect.push_back(l);
			d.clear();
			l.clear();
			if (j < ratio.size() - 1)
			{
				++j;
				sum += ratio[j];
				num = (int)(sum * n);
			}
		}
		d.push_back(data[index[i]]);
		l.push_back(labels[index[i]]);
	}
	data_sect.push_back(d);
	labels_sect.push_back(l);
	return true;
}

bool split_data(vector< vector<vector<double> > >& data_sect, vector< vector<vector<double> > >& y_sect, const vector<vector<double> >& data, const vector<vector<double> >& y, const vector<double>& ratio)
{
	int n = data.size();
	vector<int> index;
	for (int i = 0; i < n; i++)
	{
		index.push_back(i);
	}
	srand(unsigned(time(0)));
	random_shuffle(index.begin(), index.end());
	double sum = ratio[0];
	int num = sum * n;
	vector<vector<double> > d;
	vector<vector<double> > yy;
	for (int i = 0, j = 0; i < n; i++)
	{
		if (i >= num)
		{
			data_sect.push_back(d);
			y_sect.push_back(yy);
			d.clear();
			yy.clear();
			if (j < ratio.size() - 1)
			{
				++j;
				sum += ratio[j];
				num = (int)(sum * n);
			}
		}
		d.push_back(data[index[i]]);
		yy.push_back(y[index[i]]);
	}
	data_sect.push_back(d);
	y_sect.push_back(yy);
	return true;
}

bool split_data(vector<vector<double> >& data_sect, vector<int>& labels_sect, const vector<vector<double> >& data, const vector<int>& labels, const int& num)
{
	int n = data.size();
	vector<int> index;
	for (int i = 0; i < n; i++)
	{
		index.push_back(i);
	}
	srand(unsigned(time(0)));
	random_shuffle(index.begin(), index.end());
	for (int i = 0, j = 0; i < n; i++)
	{
		if (i >= num)
		{
			break;
		}
		data_sect.push_back(data[index[i]]);
		labels_sect.push_back(labels[index[i]]);
	}
	return true;
}

bool split_data(vector<vector<double> >& data_sect, vector<vector<double> >& y_sect, const vector<vector<double> >& data, const vector<vector<double> >& y, const int& num)
{
	int n = data.size();
	vector<int> index;
	for (int i = 0; i < n; i++)
	{
		index.push_back(i);
	}
	srand(unsigned(time(0)));
	random_shuffle(index.begin(), index.end());
	for (int i = 0, j = 0; i < n; i++)
	{
		if (i >= num)
		{
			break;
		}
		data_sect.push_back(data[index[i]]);
		y_sect.push_back(y[index[i]]);
	}
	return true;
}

vector<vector<double> > normalize(const vector<vector<double> >& data)
{
	vector<vector<double> > Data(data);
	vector<double> mi(data[0]), mx(data[0]);
	for (int i = 1; i < data.size(); i++)
	{
		for (int j = 0; j < data[i].size(); j++)
		{
			mi[j] = min(mi[j], data[i][j]);
			mx[j] = max(mx[j], data[i][j]);
		}
	}
	vector<vector<double> > Mi = repeat(mi, data.size(), 0), Mx = repeat(mx, data.size(), 0);
	Data = (Data - Mi) / (Mx - Mi);
	return Data;

}

#ifndef DATA
#define DATA
#include"data.h"
#include<iostream>
#include<string>
#include<vector>
using namespace std;

bool read_in(vector<vector<double> >& data, vector<int>& labels, string dataPath, string labelsPath);

bool read_in(vector<vector<double> >& data, vector<vector<double> >& y, string dataPath, string yPath);

bool split_data(vector< vector<vector<double> > >& data_sect, vector<vector<int> >& labels_sect, const vector<vector<double> >& data, const vector<int>& labels, const vector<double>& ratio);

bool split_data(vector< vector<vector<double> > >& data_sect, vector< vector<vector<double> > >& y_sect, const vector<vector<double> >& data, const vector<vector<double> >& y, const vector<double>& ratio);

bool split_data(vector< vector<vector<double> > >& data_sect, vector<vector<int> >& labels_sect, const vector<vector<double> >& data, const vector<int>& labels, const int& num);

bool split_data(vector< vector<vector<double> > >& data_sect, vector< vector<vector<double> > >& y_sect, const vector<vector<double> >& data, const vector<vector<double> >& y, const vector<int>& num);

vector<vector<double> > normalize(const vector<vector<double> >& data);

#endif

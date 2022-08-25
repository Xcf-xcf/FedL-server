#define _CRT_SECURE_NO_WARNINGS
#include"matrix.h"
#include"model.h"
#include<iostream>
#include<string>
#include<sstream>
#include<iomanip>
#include<set>
#include <algorithm>
using namespace std;

double accuracy(vector<int> true_labels, vector<int> pred_labels)
{
	double acc=0;
	set<int> cls(true_labels.begin(), true_labels.end());
	for (set<int>::iterator it = cls.begin(); it != cls.end(); it++)
	{
		acc += sum((true_labels == (*it))&&(pred_labels == (*it)));
	}
	acc /= (1.0 * true_labels.size());
	return acc;
}

vector<double> precision(vector<int> true_labels, vector<int> pred_labels)
{
	set<int> cls(true_labels.begin(), true_labels.end());
	vector<double> x;
	for (set<int>::iterator it = cls.begin(); it != cls.end(); it++)
	{
		x.push_back(1.0 * sum((true_labels == (*it)) && (pred_labels == (*it))) / sum(pred_labels == (*it)));
	}
	return x;
}

vector<double> recall(vector<int> true_labels, vector<int> pred_labels)
{
	set<int> cls(true_labels.begin(), true_labels.end());
	vector<double> x;
	for (set<int>::iterator it = cls.begin(); it != cls.end(); it++)
	{
		x.push_back(1.0 * sum((true_labels == (*it)) && (pred_labels == (*it))) / sum(true_labels == (*it)));
	}
	return x;
}

void LDA::set_para(const vector<double>& para)
{
	threshold = para[para.size() - 1];
	w.assign(para.begin(), para.end()-1);
}

vector<double> LDA::get_para()
{
	vector<double> para(w.begin(), w.end());
	para.push_back(threshold);
	return para;
}

string LDA::para_to_str(const vector<double>& para)
{
	int n = para.size();
	string str = "";
	for (int i = 0; i < n; i++)
	{
		if (i)
			str = str + " ";
		stringstream ss;
		ss << setprecision(15) << para[i];
		str = str + ss.str();
	}
	return str;
}

vector<double> LDA::str_to_para(string para)
{
	stringstream ss(para);
	double buf;
	vector<double> p;
	while (ss >> buf)
	{
		p.push_back(buf);
		//cout << buf << endl;
	}
	return p;
}

void LDA::train(const vector<vector<double> >& data, const vector<int>& labels)
{
	vector<vector<double> > d0=row(data, labels == 0), d1= row(data, labels == 1);
	vector<vector<double> > s0 = cov(d0), s1 = cov(d1);
	vector<double> m0 = mean(d0), m1 = mean(d1);
	vector<vector<double> > M0 = T(m0);
	vector<vector<double> > M1 = T(m1);
	vector<vector<double> > s = s0 + s1;
	s = inv(s + (1e-6) * eye(s.size(), s.size()));
	//cout << M0.size() << endl;
	vector<vector<double> > t = T(s * (M0 + (-1) * M1));
	w = t[0];
	M0 = T(M0);
	M1 = T(M1);
	threshold = (((0.5 * M0) + (0.5 * M1)) * T(w))[0][0];
}

vector<int> LDA::predict(const vector<vector<double> >& data)
{
	int n = data.size();
	vector<int> labels(n);
	//cout << data.size() << ' ' << data[0].size() << endl;
	//cout << w.size() << endl;
	vector<vector<double> > y = data * T(w);
	for (int i = 0; i < n; i++)
	{
		if (y[i][0] >= threshold)
		{
			labels[i] = 0;
		}
		else
		{
			labels[i] = 1;
		}
	}
	return labels;
}


BP::BP(int Epochs, int Batch_size, double Learning_rate, double Beta, int Gap, int Lr_reduced_choice, vector<layer> Layers, double (*Loss)(const vector<double>& output, const vector<double>& y), vector<double>(*Dloss)(const vector<double>& output, const vector<double>& y), const vector<vector<double> >& data, const vector<vector<double> >& y)
{
	epochs = Epochs;
	batch_size = Batch_size;
	learning_rate = Learning_rate;
	beta = Beta;
	gap = Gap;
	lr_reduced_choice = Lr_reduced_choice;
	layers.assign(Layers.begin(), Layers.end());
	loss = Loss;
	dloss = Dloss;
	n = data.size();
	srand(time(0));
	vector<int> index;
	for (int i = 0; i < data.size(); i++)
	{
		index.push_back(i);
	}
	random_shuffle(index.begin(), index.end());
	vector<vector<double> > data_sect, y_sect;
	for (int i = 0, j = 0; i < data.size(); i++)
	{
		if (j >= batch_size)
		{
			j = 0;
			data_batch.push_back(data_sect);
			y_batch.push_back(y_sect);
			data_sect.clear();
			y_sect.clear();
		}
		data_sect.push_back(data[index[i]]);
		y_sect.push_back(y[index[i]]);
	}
	data_batch.push_back(data_sect);
	y_batch.push_back(y_sect);
	batch_idx = data_batch.size();
}

bool BP::const_init(double w_const, double b_const)
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].const_init(w_const, b_const);
	}
	return true;
}

bool BP::gaussian_init(double mean, double var)
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].gaussian_init(mean, var);
	}
	return true;
}

bool BP::positive_unitball_init()
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].positive_unitball_init();
	}
	return true;
}

bool BP::uniform_init(double mi, double mx)
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].uniform_init(mi, mx);
	}
	return true;
}

bool BP::xavier_init()
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].xavier_init();
	}
	return true;
}

double BP::lr_reduced(int epoch)
{
	double lr=learning_rate;
	int t = epoch / gap;
	switch (lr_reduced_choice)
	{
	case 0:
		lr = lr * pow(beta, t);
	case 1:
		lr = lr * exp(-beta * t);
	}
	return lr;
}

vector<double> BP::train()
{
	vector<double> Loss(epochs,0);
	for (int i = 0; i < epochs; i++)
	{
		vector<int> pred_labels, true_labels;
		for (int j = 0; j < batch_idx; j++)
		{
			vector<vector<vector<double> > > dw(layers.size());
			vector<vector<double> > db(layers.size());
			for (int k = 0; k < data_batch[j].size(); k++)
			{
				vector<vector<double> > y;
				y.push_back(data_batch[j][k]);
				for (int l = 0; l < layers.size(); l++)
				{
					y.push_back(layers[l].y(y[l]));
				}
				vector<double> dy = dloss(y[layers.size()], y_batch[j][k]);
				//pred_labels.push_back(argmax(y[layers.size()]));
				//true_labels.push_back(argmax(y_batch[j][k]));
				//cout << dy[0] << endl;
				//cout << y[layers.size()].size() << ' ' << y[0].size() << ' ' << y[1].size() << endl;
				Loss[i] += loss(y[layers.size()], y_batch[j][k]);
				for (int l = layers.size() - 1; l >= 0; l--)
				{
					//cout << 1 << endl;
					if (j == 0)
					{
						dw[l] = layers[l].dw(y[l], dy);
						db[l] = layers[l].db(y[l], dy);
					}
					else
					{
						dw[l] = dw[l] + layers[l].dw(y[l], dy);
						db[l] = db[l] + layers[l].db(y[l], dy);
					}
					dy = layers[l].dx(y[l], dy);
				}
			}
			double lr = lr_reduced(i);
			for (int l = 0; l < layers.size(); l++)
			{
				dw[l] = dw[l] * (-lr / data_batch[j].size());
				db[l] = db[l] * (-lr / data_batch[j].size());
				//cout << dw[l][0][0] << ' ' << dw[l][0][1] << ' ' << dw[l][0][2] << ' ' << db[l][0] << endl;
				layers[l].set_w(layers[l].get_w() + dw[l]);
				layers[l].set_b(layers[l].get_b() + db[l]);
			}
		}
		Loss[i] /= n;
		vector<vector<double> > pred = predict(test_data);
		double test_loss = MSE(pred, test_y);
		cout << "Epoch " << i + 1 << ": " << test_loss << endl;
		FILE* file = fopen("result.txt", "a+");
		stringstream ss;
		ss << setprecision(15) << test_loss;
		ss << "\n";
		fputs(ss.str().c_str(), file);
		fclose(file);
		//printf("accuarcy: %5.2lf%%\n", accuracy(true_labels, pred_labels) * 100);
		//cout << para_to_str(get_para()) << endl;
	}
	return Loss;
}

vector<vector<double> > BP::predict(vector<vector<double> > data)
{
	vector<vector<double> > y;
	for (int i = 0; i < data.size(); i++)
	{
		vector<double> x = data[i];
		for (int j = 0; j < layers.size(); j++)
		{
			x = layers[j].y(x);
		}
		y.push_back(x);
	}
	return y;
}

vector<vector<vector<double> > > BP::get_para()
{
	vector<vector<vector<double> > > para;
	for (int i = 0; i < layers.size(); i++)
	{
		vector<vector<double> > w = layers[i].get_w();
		vector<double> b = layers[i].get_b();
		for (int j = 0; j < b.size(); j++)
		{
			w[j].push_back(b[j]);
		}
		para.push_back(w);
	}
	return para;
}

bool BP::set_para(const vector<vector<vector<double> > >& para)
{
	for (int i = 0; i < layers.size(); i++)
	{
		vector<vector<double> > w(para[i].size());
		vector<double> b(para[i].size());
		for (int j = 0; j < para[i].size(); j++)
		{
			w[j].assign(para[i][j].begin(), para[i][j].end()-1);
			b.push_back(para[i][j][para[i][j].size() - 1]);
		}
		layers[i].set_w(w);
		layers[i].set_b(b);
	}
	return true;
}

vector<vector<vector<double> > > BP::str_to_para(const string& str)
{
	vector<vector<vector<double> > > para;
	stringstream sss(str);
	string layer = "", buf = "";
	while (getline(sss, layer, ';'))
	{
		vector<vector<double> > p;
		stringstream ss(layer);
		while (getline(ss, buf))
		{
			vector<double> x;
			stringstream s(buf);
			double cnt;
			while (s >> cnt)
			{
				x.push_back(cnt);
			}
			p.push_back(x);
		}
		para.push_back(p);
	}
	return para;
	
}

string BP::para_to_str(const vector<vector<vector<double> > >& para)
{
	string str = "";
	for (int i = 0; i < para.size(); i++)
	{
		if (i)
			str += ";";
		for (int j = 0; j < para[i].size(); j++)
		{
			if (j)
				str += "\n";
			for (int k = 0; k < para[i][j].size(); k++)
			{
				if (k)
					str += " ";
				stringstream ss;
				ss << setprecision(15) << para[i][j][k];
				str += ss.str();
			}
		}
	}
	return str;
}
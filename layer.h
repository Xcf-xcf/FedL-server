#ifndef LAYER
#define LAYER
#include<iostream>
#include<vector>
#include"matrix.h"
using namespace std;

class layer
{
private:
	int n, m;                            //n为输入维数，m为输出维数
	vector<vector<double> > w;           //权值
	vector<double> b;                    //偏置项
	vector<double> (*f)(const vector<double>&); //激活函数
	vector<double> (*df)(const vector<double>&);//激活函数导数
public:
	layer(int N, int M, vector<double> (*F)(const vector<double>&), vector<double> (*Df)(const vector<double>&));
	//五种参数初始化方法
	bool const_init(double w_const, double b_const);
	bool gaussian_init(double mean, double var);
	bool positive_unitball_init();
	bool uniform_init(double mi, double mx);
	bool xavier_init();
	//获取参数
	vector<vector<double> > get_w();
	vector<double> get_b();
	//设置参数
	bool set_w(const vector<vector<double> >& W);
	bool set_b(const vector<double>& B);
	//求该层输出
	vector<double> y(const vector<double>& x);
	//求loss对该层输入的偏导
	vector<double> dx(const vector<double>& x, const vector<double>& dy);
	//求loss对该层参数的偏导
	vector<vector<double> > dw(const vector<double>& x, const vector<double>& dy);
	vector<double> db(const vector<double>& x, const vector<double>& dy);
};

#endif // !LAYER


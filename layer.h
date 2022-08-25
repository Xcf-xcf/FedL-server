#ifndef LAYER
#define LAYER
#include<iostream>
#include<vector>
#include"matrix.h"
using namespace std;

class layer
{
private:
	int n, m;                            //nΪ����ά����mΪ���ά��
	vector<vector<double> > w;           //Ȩֵ
	vector<double> b;                    //ƫ����
	vector<double> (*f)(const vector<double>&); //�����
	vector<double> (*df)(const vector<double>&);//���������
public:
	layer(int N, int M, vector<double> (*F)(const vector<double>&), vector<double> (*Df)(const vector<double>&));
	//���ֲ�����ʼ������
	bool const_init(double w_const, double b_const);
	bool gaussian_init(double mean, double var);
	bool positive_unitball_init();
	bool uniform_init(double mi, double mx);
	bool xavier_init();
	//��ȡ����
	vector<vector<double> > get_w();
	vector<double> get_b();
	//���ò���
	bool set_w(const vector<vector<double> >& W);
	bool set_b(const vector<double>& B);
	//��ò����
	vector<double> y(const vector<double>& x);
	//��loss�Ըò������ƫ��
	vector<double> dx(const vector<double>& x, const vector<double>& dy);
	//��loss�Ըò������ƫ��
	vector<vector<double> > dw(const vector<double>& x, const vector<double>& dy);
	vector<double> db(const vector<double>& x, const vector<double>& dy);
};

#endif // !LAYER


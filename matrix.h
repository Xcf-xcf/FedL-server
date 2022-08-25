#ifndef MATRIX
#define MATRIX
#include<iostream>
#include<vector>
using namespace std;

vector<vector<double> > eye(const int& n, const int& m);                  //��λ��

vector<vector<double> > diag(const vector<double>& a);             //���Խ���Ԫ��Ϊa��Ԫ�صķ���

vector<vector<double> > T(const vector<vector<double> >& a);      //ת��

vector<vector<double> > T(const vector<double>& a);               

vector<double> flat(const vector<vector<double> >& a);            //������չ��

vector<vector<double> > repeat(const vector<double>& a, const int& n, const int& axis=0);//ÿ�У�axis=0����ÿ�У�axis=1����Ϊa��Ԫ�صľ���

double det(vector<vector<double> > a);                                             //����ʽ

vector<vector<double> > operator*(const vector<vector<double> >& a, const vector<vector<double> >& b);//���������

vector<vector<double> > operator*(const vector<vector<double> >& a, const double& b);                 //�����������

vector<vector<double> > operator*(const double& b, const vector<vector<double> >& a);

vector<double> operator*(const vector<double>& a, const double& b);

vector<double> operator*(const double& b, const vector<double>& a);

vector<double> operator*(const vector<double>& a, const vector<double>& b);                           

vector<vector<double> > operator/(const vector<vector<double> >& a, const vector<vector<double> >& b);

vector<vector<double> > operator+(const vector<vector<double> >& a, const vector<vector<double> >& b);//����ӷ�

vector<vector<double> > operator+(const vector<vector<double> >& a, const double& b);

vector<vector<double> > operator+(const double& b, const vector<vector<double> >& a);

vector<double> operator+(const vector<double>& a, const vector<double>& b);

vector<double> operator+(const vector<double>& a, const double& b);

vector<double> operator+(const double& b, const vector<double>& a);

vector<vector<double> > operator-(const vector<vector<double> >& a, const vector<vector<double> >& b);

vector<double> operator-(const vector<double>& a, const vector<double>& b);

vector<double> operator-(const vector<double>& a, const double& b);

vector<double> operator-(const double& b, const vector<double>& a);



template<typename T>
vector<bool> operator==(const vector<T>& a, const T& b)
{
	int n = a.size();
	vector<bool> x(n);
	for (int i = 0; i < n; i++)
	{
		x[i] = (a[i] == b);
	}
	return x;
}

vector<bool> operator&&(const vector<bool>& a, const vector<bool>& b);

int sum(const vector<bool>& a);

vector<vector<double> > row(const vector<vector<double> >& a, const vector<bool>& b);

vector<vector<double> > col(const vector<vector<double> >& a, const vector<bool>& b);

vector<int> argmax(const vector<vector<double> >& a, const int& axis);

int argmax(const vector<double>& a);

vector<vector<double> > inv(const vector<vector<double> >& a);                         //��������

vector<double> mean(const vector<vector<double> >& a, int axis=0);                     //���ֵ��axis=0�������󣬷�֮������

vector<vector<double> > cov(const vector<vector<double> >& a, bool colvar = true);//�󷽲colvar=true����a��ÿ��Ϊһ����������֮ÿ��Ϊһ��������

vector<double> Y(const vector<double>& x);           // y=x

vector<double> Dy(const vector<double>& x);         // dy=1

vector<double> sigmoid(const vector<double>& x);    // y=1/(1+e^(-x))

vector<double> dsigmoid(const vector<double>& x);   // dy=y*(1-y)

vector<double> softmax(const vector<double>& x);    // yi=e^(xi)/sum(e^(xj)),j=1,2...n (nΪ�������

vector<double> dsoftmax(const vector<double>& x);   

vector<double> relu(const vector<double>& x);      // y=x (x>=0), 0 (x<0)

vector<double> drelu(const vector<double>& x);     // dy=1 (x>0), 0(x<=0)

double MSE(const vector<double>& y, const vector<double>& Y);  //L2_loss

double MSE(const vector<vector<double> >& y, const vector<vector<double> >& Y);

vector<double> dMSE(const vector<double>& y, const vector<double>& Y);

double CrossEntropy(const vector<double>& y, const vector<double>& Y);    //������

double CrossEntropy(const vector<vector<double> >& y, const vector<vector<double> >& Y);

vector<double> dCrossEntropy(const vector<double>& y, const vector<double>& Y);

double gaussrand();                   //0-1��˹�ֲ�����

#endif

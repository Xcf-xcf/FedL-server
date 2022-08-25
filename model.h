#ifndef MODEL
#define MODEL
#include<iostream>
#include<vector>
#include"matrix.h"
#include"layer.h"

double accuracy(vector<int> true_labels, vector<int> pred_labels);//��ȷ��

vector<double> precision(vector<int> true_labels, vector<int> pred_labels);//��׼��

vector<double> recall(vector<int> true_labels, vector<int> pred_labels);//��ȫ��

double ROC(vector<int> true_labels, vector<int> pred_labels);//ROC����

class LDA//������
{
private:
	vector<double> w;       //ͶӰ������ģ�Ͳ���
	double threshold;       //������ֵ
public:
	LDA(int d) { w.resize(d); threshold = 0; }                                   //dά����
	void set_para(const vector<double>& para);                   //����ģ�Ͳ���
	vector<double> get_para();                                   //��ȡʵ������ģ�Ͳ���
	string para_to_str(const vector<double>& para);              //��ʵ�ʲ���ת��Ϊ����ͨ�ŵ��ı����Ͳ���
	vector<double> str_to_para(string para);                     //������ͨ�ŵ��ı����Ͳ���ת��Ϊʵ�ʲ���
	void train(const vector<vector<double> >& data, const vector<int>& labels);//ѵ��ģ��
	vector<int> predict(const vector<vector<double> >& data);           //���Ԥ����
};

class BP
{
private:
	int epochs;
	int batch_size, batch_idx;
	int n;
	double learning_rate, beta;
	int gap, lr_reduced_choice;
	vector<layer> layers;
	double (*loss)(const vector<double>& output, const vector<double>& y);
	vector<double> (*dloss)(const vector<double>& output, const vector<double>& y);
	vector<vector<vector<double> > > data_batch;
	vector<vector<vector<double> > > y_batch;
	vector<vector<double> > test_data;
	vector<vector<double> > test_y;
public:
	BP(int Epochs, int Batch_size, double Learning_rate, double Beta, int Gap, int Lr_reduced_choice, vector<layer> Layers, double (*Loss)(const vector<double>& output, const vector<double>& y), vector<double> (*Dloss)(const vector<double>& output, const vector<double>& y), const vector<vector<double> >& data, const vector<vector<double> >& y);
	bool set_test(vector<vector<double> > Test_data, vector<vector<double> > Test_y)
	{
		test_data.assign(Test_data.begin(), Test_data.end());
		test_y.assign(Test_y.begin(), Test_y.end());
		return true;
	}
	bool const_init(double w_const=0, double b_const=0);
	bool gaussian_init(double mean=0, double var=1);
	bool positive_unitball_init();
	bool uniform_init(double mi=0, double mx=1);
	bool xavier_init();
	double lr_reduced(int epoch);
	vector<double> train();
	vector<vector<double> > predict(vector<vector<double> > data);
	vector<vector<vector<double> > > get_para();
	bool set_para(const vector<vector<vector<double> > >& para);
	vector<vector<vector<double> > > str_to_para(const string& str);
	string para_to_str(const vector<vector<vector<double> > >& para);
};
#endif

#include <stdio.h>
#include <time.h>

float computeCost(float *x, float *y, float *theta, int m) {
	// compute cost for linear regression

	// initialize values
	int i;
	float J = 0, sigma = 0;
	
	// compute cost using the formula given
	for (i = 0; i < m; i++) {
		sigma += ((theta[0] + theta[1] * x[i]) - y[i]) * ((theta[0] + theta[1] * x[i]) - y[i]);
	}

	J = sigma / (2 * m);
	return J;
}

void gradientDescent(float *x, float *y, float *theta, float alpha, int num_iters, int m) {
	// performs gradient descent to learn theta

	// initialize values
	int i, j;
	float sigma[2] = {0, 0}, cost;

	for (j = 0; j < num_iters; j++) {
		for (i = 0; i < m; i++) {
			sigma[0] += ((theta[0] + theta[1] * x[i]) - y[i]);
			sigma[1] += ((theta[0] + theta[1] * x[i]) - y[i]) * x[i];
		}

		// compute new theta
		theta[0] = theta[0] - (alpha / m) * sigma[0];
		theta[1] = theta[1] - (alpha / m) * sigma[1];

		// reset sigma
		sigma[0] = 0;
		sigma[1] = 0;

		// list cost to see its cnovergence
		cost = computeCost(x, y, theta, m);
		printf("%d: Cost = %f Theta = %f %f\n", j + 1, cost, theta[0], theta[1]);
	}

	return;

}

long timediff(clock_t t1, clock_t t2) {
	long elapsed;
	elapsed = (double)(t2 - t1) / CLOCKS_PER_SEC * 1000;
	return elapsed;
}

int main() {
	// load input data
	FILE *fp;
	fp = fopen("ex1data1.txt", "r");
	if (fp == NULL) {
		printf("Couldn't open file\n");
		return 1;
	}

	float x[97], y[97];
	int i;

	for (i = 0; i < 97; i++) {
		fscanf(fp, "%f,%f", &x[i], &y[i]);
	}

	fclose(fp);

	int size = sizeof(y) / sizeof(float);
	printf("%d\n", size);

	// compute initial cost for testing
	float test;
	float theta[2] = {0, 0};
	test = computeCost(x, y, theta, size);
	printf("Initial Cost = %f\n", test);

	// compute optimal theta
	float alpha = 0.01;
	int num_iters = 1500;
	clock_t start, end;

	start = clock();
	gradientDescent(x, y, theta, alpha, num_iters, size);
	end = clock();

	printf("Elapsed time is %lu ms.\n", timediff(start, end));

	return 0;
}

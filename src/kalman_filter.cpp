#include "kalman_filter.h"
#include <iostream>

#define PI 3.14159265

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in,
                        float noise_ax, float noise_ay) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  noise_ax_ = noise_ax;
  noise_ay_ = noise_ay;
}

void KalmanFilter::Predict(float dt) {

  /*
   * Intermediate values
   */
  float dt44 = (float) (dt * dt * dt * dt / 4.0);
  float dt32 = (float) (dt * dt * dt / 2.0);
  float dt2 = dt*dt;

  /*
   * Update state transition matrix with time delta
   */
  F_(0, 2) = dt;
  F_(1, 3) = dt;

  Q_ <<
      dt44*noise_ax_, 0, dt32*noise_ax_, 0,
      0, dt44*noise_ay_, 0, dt32*noise_ay_,
      dt32*noise_ax_, 0, dt2*noise_ax_, 0,
      0, dt32*noise_ay_, 0, dt2*noise_ay_;

  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_*x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  VectorXd hx(3);
  float d = (float) sqrt(x_[0] * x_[0] + x_[1] * x_[1]);
  float phi = atan2(x_[1], x_[0]);
  float radial_velocity = (x_[0]*x_[2]+x_[1]*x_[3])/d;

  hx <<
      d,
      phi,
      radial_velocity;

  VectorXd y = z - hx;

  /*
   * Normalize the angle between -PI and PI
   */
  float y_phi = y[1];
  y_phi = atan2(sin(y_phi), cos(y_phi));
  y(1) = y_phi;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

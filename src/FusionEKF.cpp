#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  H_ = MatrixXd(2, 4);
  H_ <<
    1, 0, 0, 0,
    0, 1, 0, 0;

  //measurement covariance matrix - laser
  R_laser_ <<
      0.0225, 0,
      0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ <<
      0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    VectorXd x(4);

    /*
     * Initial process covariance matrix
     */
    MatrixXd P(4, 4);
    P <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    /*
     * Initial state transition matrix
     */
    MatrixXd F(4, 4);
    F <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    /*
     * Initial process covariance matrix
     */
    MatrixXd Q(4, 4);
    Q <<
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float r = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];

      x <<  r * cos(phi), r * sin(phi), 0, 0;

      ekf_.Init(x, P, F, Hj_, R_radar_, Q, 9.0, 9.0);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);

      x << px, py, 0, 0;

      /**
      Initialize state.
      */
      ekf_.Init(x, P, F, H_, R_laser_, Q, 9.0, 9.0);
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (float) ((measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0);	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.Predict(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_;

    ekf_.Update(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_radar_;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

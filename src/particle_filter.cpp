/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#include <assert.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#define Err 0.00001
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if (is_initialized) {
    return;
  }
  default_random_engine gen;

  num_particles = 100;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  // Creating normal distributions
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Calculate new state.
  for (auto it = particles.begin(); it != particles.end(); ++it) {

  // Check if Yaw is too small
  if ( fabs(yaw_rate) < Err ) {
      it->x += velocity * delta_t * cos( it->theta );
      it->y += velocity * delta_t * sin( it->theta );
    } else {
      it->x += velocity / yaw_rate * ( sin( it->theta + yaw_rate * delta_t ) - sin( it->theta ) );
      it->y += velocity / yaw_rate * ( cos( it->theta ) - cos( it->theta + yaw_rate * delta_t ) );
      it->theta += yaw_rate * delta_t;
    }

    it->x += dist_x(gen);
    it->y += dist_y(gen);
    it->theta += dist_theta(gen);
  }

}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    //
    //   Assign observation id to predicted
    for (auto it = observations.begin(); it != observations.end(); ++it) {
      double max_dist = 10000000.0;
      for (auto itin = predicted.begin(); itin != predicted.end(); ++itin) {
        double d = dist(it->x, it->y, itin->x, itin->y);
        if (d < max_dist) {
          max_dist = d;
          it->id = itin->id;
        }
      }
    }
}

inline bool compareById(const LandmarkObs &a, const LandmarkObs &b) {
  return a.id < b.id;
}

inline double getWeight(const LandmarkObs &obs, const LandmarkObs &pred, double std_landmark[]) {
  double helper = (((obs.x - pred.x) * (obs.x - pred.x)) / (2.0 * std_landmark[0] * std_landmark[0])) +
                   (((obs.y - pred.y) * (obs.y - pred.y)) / (2.0 * std_landmark[1] * std_landmark[1]));
  double hel2 = exp(-1 * helper);
  return (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1])) * hel2;
}

inline std::vector<LandmarkObs> transform(const std::vector<LandmarkObs> &observations,
                                           const Particle &p) {
  std::vector<LandmarkObs> transformed;
  for (auto it = observations.begin(); it != observations.end(); ++it) {
    LandmarkObs lo;
    lo.id = it->id;
    lo.x  = p.x + (it->x * cos(p.theta)) - (it->y * sin(p.theta));
    lo.y  = p.y + (it->x * sin(p.theta)) + (it->y * cos(p.theta));
    transformed.push_back(lo);
  }
  return transformed;
}

inline std::vector<LandmarkObs> predictLandmarks(const Particle &p, double sensor_range,
                                            const Map &map_landmarks) {
  std::vector<LandmarkObs> predictions;
  for (auto it = map_landmarks.landmark_list.begin(); 
       it != map_landmarks.landmark_list.end(); ++it) {
    double dis = dist(p.x, p.y, it->x_f, it->y_f);
    if (dis <= (sensor_range)) {
      LandmarkObs lo;
      lo.id = it->id_i;
      lo.x = it->x_f;
      lo.y = it->y_f;
      predictions.push_back(lo);
    }
  }
  return predictions;
}

inline LandmarkObs getMatchingPred(std::vector<LandmarkObs> predicted, long id) {
  for (int i = 0; i < predicted.size(); i++) {
    if (predicted.at(i).id == id)
      return predicted.at(i);
  }
  std::cout << "Should not be here..." << std::endl;
  assert(false);
  LandmarkObs ob;
  return ob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double sum = 0.0;
    for (auto it = particles.begin(); it != particles.end(); ++it) {
    // Find landmarks in particle's range.
    std::vector<LandmarkObs> predicted = predictLandmarks(*it, sensor_range, map_landmarks);
    // Transform observation coordinates.
    std::vector<LandmarkObs> transformed_observations = transform(observations, *it);

      //  --- use data association
    dataAssociation(predicted, transformed_observations);
    double weight = 1.0;
    for (int i = 0; i < transformed_observations.size(); i++) {
      double tw = getWeight(transformed_observations.at(i), 
                            getMatchingPred(predicted, transformed_observations.at(i).id),
                            std_landmark);
      tw = (tw <= 0.00000001)? Err : tw;
      weight *= tw;
    }
    it->weight = weight;
    sum += weight;
  }
  //  --- normalize these weights
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    it->weight = it->weight/sum;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<int> dist_nums;
  std::vector<Particle> particles_new;
  std::random_device rd;
  std::mt19937 gen(rd());
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    dist_nums.push_back((int)(it->weight * 1000000));
  }
  std::discrete_distribution<> d(dist_nums.begin(), dist_nums.end());
  for(int n=0; n<particles.size(); ++n) {
    particles_new.push_back(particles.at(d(gen)));
  }
  particles = particles_new;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

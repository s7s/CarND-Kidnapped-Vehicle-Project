/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles=25;
	particles.resize(num_particles);
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> head_theta(theta, std[2]);
	for(int i=0;i<num_particles;i++){
		particles[i].id=i;
		particles[i].x=dist_x(gen);
		particles[i].y=dist_y(gen);
		particles[i].theta=head_theta(gen);
		particles[i].weight=1;

	}
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for(int i=0;i<num_particles;i++){

    if(yaw_rate==0){
      particles[i].x=particles[i].x+velocity*cos(particles[i].theta)*delta_t;
      particles[i].y=particles[i].y+velocity*sin(particles[i].theta)*delta_t;
    }
    else{
      particles[i].x=particles[i].x+(velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
      particles[i].y=particles[i].y+(velocity/yaw_rate)*(-cos(particles[i].theta+yaw_rate*delta_t)+cos(particles[i].theta));
    }
    particles[i].theta=particles[i].theta+yaw_rate*delta_t;

		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> head_theta(particles[i].theta, std_pos[2]);

		particles[i].x=dist_x(gen);
		particles[i].y=dist_y(gen);
		particles[i].theta=head_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0;i<num_particles;i++){
		for(int j=0;j<observations.size();j++){
			// transform to map x coordinate
			particles[i].sense_x.push_back( particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y));

			// transform to map y coordinate
			particles[i].sense_y.push_back( particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y));
			double temp_distance;
			double distance;
			for(int k=0;k<predicted.size();k++){
				temp_distance=dist(particles[i].sense_x[j], particles[i].sense_y[j], predicted[k].x, predicted[k].y);
				if(k==0){
					particles[i].associations.push_back(k);
					distance=temp_distance;
				}
				else{
					if(temp_distance<distance){
						distance=temp_distance;
						particles[i].associations[j]=k;
					}
				}

			}	
		}
	}
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
	std::vector<LandmarkObs> sensed_observations=observations;
	int index=0;
	for(int j=0;j<observations.size();j++){
		if (dist(observations[j].x, observations[j].y, 0, 0)>sensor_range){
			sensed_observations.erase (sensed_observations.begin()+j-index);
			index++;		
		}
	}
	std::vector<LandmarkObs> predicted;
	predicted.resize(map_landmarks.landmark_list.size());
	for(int i=0;i<map_landmarks.landmark_list.size();i++){
		predicted[i].id=map_landmarks.landmark_list[i].id_i;
		predicted[i].x=map_landmarks.landmark_list[i].x_f;
		predicted[i].y=map_landmarks.landmark_list[i].y_f;
	}

	dataAssociation( predicted, sensed_observations);
	for(int i=0;i<num_particles;i++){
		particles[i].weight=1.0;
		for(int j=0;j<sensed_observations.size();j++){
			particles[i].weight*=(1/(2*M_PI*std_landmark[0]*std_landmark[1]))*exp(-((pow(particles[i].sense_x[j]-predicted[particles[i].associations[j]].x,2)/(2*std_landmark[0]*std_landmark[0]))+(pow(particles[i].sense_y[j]-predicted[particles[i].associations[j]].y,2)/(2*std_landmark[1]*std_landmark[1]))));
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	float weights_sum=0;
	float weights_max=0;
	for(int i=0;i<num_particles;i++){
		weights_sum+=particles[i].weight;
	}
	for(int i=0;i<num_particles;i++){
		//cout<<particles[i].weight<<endl;
		particles[i].weight/=weights_sum;
		if(particles[i].weight>weights_max){
			weights_max=particles[i].weight;
		}
	}
	//cout<<"sum= "<<weights_sum<<endl;
	//cout<<"max= "<<weights_max<<endl;
	std::default_random_engine float_generator;
  std::uniform_real_distribution<double> distribution_real(0.0,2*weights_max);
	std::default_random_engine int_generator;
  std::uniform_int_distribution<int> distribution_int(0,num_particles-1);

  int index = distribution_int(int_generator);
  double increase;
	std::vector<Particle> new_particles;
	new_particles.resize(num_particles);

  for(int i=0;i<num_particles;i++){
  	increase= distribution_real(float_generator);
  	while(increase>particles[index].weight){
  		increase-=particles[index].weight;
  		index+=1;
  		index%=num_particles;
  	}
  	new_particles[i].id=i;
  	new_particles[i].x=particles[index].x;
  	new_particles[i].y=particles[index].y;
  	new_particles[i].theta=particles[index].theta;
  	new_particles[i].weight=particles[index].weight;

  }
  particles=new_particles;
 
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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

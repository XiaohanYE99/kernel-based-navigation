/*
 * Circle.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <https://gamma.cs.unc.edu/RVO2/>
 */

/*
 * Example file showing a demo with 250 agents initially positioned evenly
 * distributed on a circle attempting to move to the antipodal position on the
 * circle.
 */




#ifndef RVO_OUTPUT_TIME_AND_POSITIONS
#define RVO_OUTPUT_TIME_AND_POSITIONS 1
#endif

#include <cmath>
#include <cstddef>
#include <vector>
#include <stdlib.h>

#if RVO_OUTPUT_TIME_AND_POSITIONS
#include <iostream>
#endif

#if _OPENMP
#include <omp.h>
#endif
#include <Eigen/Dense>
#include <GL/glut.h>

#include <RVO.h>
//#define TEST
#ifndef M_PI
const double M_PI = 3.14159265358979323846f;
#endif
const static int WINDOW_WIDTH = 1000;
const static int WINDOW_HEIGHT = 1000;
const static double VIEW_WIDTH = 500;
const static double VIEW_HEIGHT = 500;
/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;
RVO::RVOSimulator *sim = new RVO::RVOSimulator();
void InitGL(void)
{
	glClearColor(0.9f,0.9f,0.9f,1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(10.0);
	glMatrixMode(GL_PROJECTION);
}
void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);
	glColor4f(0.2f, 0.6f, 1.0f, 1);
	glBegin(GL_POINTS);
	for (size_t i = 0; i < sim->getNumAgents()/2; ++i) {
		glVertex2f(250+sim->getAgentPosition(i).x(),250+sim->getAgentPosition(i).y());
	}
	glEnd();
	glColor4f(0.8f, 0.6f, 1.0f, 1);
	glBegin(GL_POINTS);
	for (size_t i = sim->getNumAgents()/2; i < sim->getNumAgents(); ++i) {
		glVertex2f(250+sim->getAgentPosition(i).x(),250+sim->getAgentPosition(i).y());
	}
	glEnd();
    glFlush();
	glutSwapBuffers();

}

void setupScenario(RVO::RVOSimulator *sim)
{
	/* Specify the global time step of the simulation. */
	sim->setTimeStep(0.5f);
    sim->setNewtonParameters(100,1e-2,50,1e0,1e-6);
	/* Specify the default parameters for agents that are subsequently added. */
	sim->setAgentDefaults(15.0f, 100, 10.0, 10.0, 1.5f, 2.0f);

	/*
	 * Add agents, specifying their start position, and store their goals on the
	 * opposite side of the environment.
	 */
	for (size_t i = 0; i < 100; ++i) {
		sim->addAgent(200.0 *
		              RVO::Vector2(std::cos(i * 2.0f * M_PI / 100.0),
		                           std::sin(i * 2.0f * M_PI / 100.0))
								   +0.001*RVO::Vector2(rand()%1000,rand()%1000));
		goals.push_back(-sim->getAgentPosition(i));
	}
}

#if RVO_OUTPUT_TIME_AND_POSITIONS
void updateVisualization(RVO::RVOSimulator *sim)
{
	/* Output the current global time. */
	std::cout << sim->getGlobalTime();

	/* Output the current position of all the agents. */
	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		std::cout << " " << sim->getAgentPosition(i);
	}

	std::cout << std::endl;
}
#endif

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
	/*
	 * Set the preferred velocity to be a vector of unit magnitude (speed) in the
	 * direction of the goal.
	 */
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {
		RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);

		if (RVO::absSq(goalVector) > 1.0f) {
			goalVector = RVO::normalize(goalVector);
		}

		sim->setAgentPrefVelocity(i, goalVector);
	}
}

bool reachedGoal(RVO::RVOSimulator *sim)
{
	/* Check if all agents have reached their goals. */
	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		if (RVO::absSq(sim->getAgentPosition(i) - goals[i]) > sim->getAgentRadius(i) * sim->getAgentRadius(i)) {
			return false;
		}
	}

	return true;
}
void Update(void)
{
    #if RVO_OUTPUT_TIME_AND_POSITIONS
		//updateVisualization(sim);
	#endif
		setPreferredVelocities(sim);
	#ifdef TEST
		sim->checkEnergyFD();
	#else
		sim->doStep();
	#endif	
		glutPostRedisplay();
}
int main(int argc, char **argv)
{
	/* Create a new simulator instance. */
	/* Set up the scenario. */
	setupScenario(sim);

	/* Perform (and manipulate) the simulation. */
	/*do {

	}
	while (!reachedGoal(sim));*/
    glutInitWindowSize(WINDOW_WIDTH,WINDOW_HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow("SPH");
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
    InitGL();
	glutMainLoop();
	delete sim;

	return 0;
}

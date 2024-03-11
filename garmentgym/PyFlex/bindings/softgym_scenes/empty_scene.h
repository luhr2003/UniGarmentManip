#pragma once
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
namespace py=pybind11;

class EmptyScene : public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;

    EmptyScene(const char *name) : Scene(name) {}

    void Initialize_empty(py::dict scene_params)
    {
        g_drawPoints = false;
        g_drawCloth = false;
        g_drawSprings = false;
        for (auto item : scene_params){
            string key = py::str(item.first);
            if (key == "radius") g_params.radius = std::stof(py::str(item.second));
            if (key == "buoyancy") g_params.buoyancy = std::stof(py::str(item.second));
            if (key == "collisionDistance") g_params.collisionDistance = std::stoi(py::str(item.second));

            if (key == "numExtraParticles") g_numExtraParticles = std::stoi(py::str(item.second));
        }

        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_params.dynamicFriction = 0.75f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;

        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);
    }
};
/*
 * chomp_planning_context.cpp
 *
 *  Created on: 27-Jul-2016
 *      Author: ace
 */

#include <chomp_interface/chomp_planning_context.h>
#include <moveit_msgs/CollisionObject.h>
#include <shape_msgs/SolidPrimitive.h>
// #include <moveit_msgs/PlanningScene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

namespace chomp_interface
{
CHOMPPlanningContext::CHOMPPlanningContext(const std::string& name, const std::string& group,
                                           const robot_model::RobotModelConstPtr& model)
  : planning_interface::PlanningContext(name, group), robot_model_(model)
{
  chomp_interface_ = CHOMPInterfacePtr(new CHOMPInterface(group));

  boost::shared_ptr<collision_detection::CollisionDetectorAllocator> hybrid_cd(
      collision_detection::CollisionDetectorAllocatorHybrid::create());

  planning_scene::PlanningScenePtr planning_scene_ptr;
  if (!this->getPlanningScene())
  {
    ROS_INFO_STREAM("Configuring New Planning Scene.");
    planning_scene_ptr = planning_scene::PlanningScenePtr(new planning_scene::PlanningScene(model));
    planning_scene_ptr->setActiveCollisionDetector(hybrid_cd, true);
    setPlanningScene(planning_scene_ptr);
  }

  // we changed this
  // create our own planning scene
  tf_ = boost::shared_ptr<tf::Transformer>(new tf::Transformer());
  psm_ = planning_scene_monitor::PlanningSceneMonitorPtr(new planning_scene_monitor::PlanningSceneMonitor(planning_scene_ptr, "robot_description", tf_));
  psm_->startSceneMonitor();
  psm_->startWorldGeometryMonitor();
  psm_->startStateMonitor();
}

CHOMPPlanningContext::~CHOMPPlanningContext()
{
}

bool CHOMPPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res)
{
  moveit_msgs::MotionPlanDetailedResponse res2;
  
  // moveit_msgs::PlanningScene sceneMsg;
  // planning_scene_->getPlanningSceneMsg(sceneMsg);
  // std::cout << sceneMsg << "scene message: " << std::endl;
  planning_scene_->printKnownObjects(std::cout);

  if (chomp_interface_->solve(planning_scene_, request_, chomp_interface_->getParams(), res2))
  {
    res.trajectory_.resize(1);
    res.trajectory_[0] =
        robot_trajectory::RobotTrajectoryPtr(new robot_trajectory::RobotTrajectory(robot_model_, getGroupName()));

    moveit::core::RobotState start_state(robot_model_);
    robot_state::robotStateMsgToRobotState(res2.trajectory_start, start_state);
    res.trajectory_[0]->setRobotTrajectoryMsg(start_state, res2.trajectory[0]);

    trajectory_processing::IterativeParabolicTimeParameterization itp;
    itp.computeTimeStamps(*res.trajectory_[0], request_.max_velocity_scaling_factor,
                          request_.max_acceleration_scaling_factor);

    res.description_.push_back("plan");
    res.processing_time_ = res2.processing_time;
    res.error_code_ = res2.error_code;
    return true;
  }
  else
  {
    res.error_code_ = res2.error_code;
    return false;
  }
}

bool CHOMPPlanningContext::solve(planning_interface::MotionPlanResponse& res)
{
  planning_interface::MotionPlanDetailedResponse res_detailed;
  bool result = solve(res_detailed);

  res.error_code_ = res_detailed.error_code_;
  res.trajectory_ = res_detailed.trajectory_[0];
  res.planning_time_ = res_detailed.processing_time_[0];

  return result;
}

bool CHOMPPlanningContext::terminate()
{
  // TODO - make interruptible
  return true;
}

void CHOMPPlanningContext::clear()
{
}

} /* namespace chomp_interface */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_representation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <std_msgs/String.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <string>
//#include <Eigen/Core>

//typedefs
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// publisher

ros::Publisher detectionOutcome;

// subscriber

ros::Subscriber sub;
ros::Subscriber enableDetection;

// viewer window
pcl::visualization::PCLVisualizer viewer("PCL Viewer");

// global variables
bool isFirstPC;
bool pointsSelected;
bool gotPointCloud;
bool startDetection;
//
PointCloud testCloud;
const sensor_msgs::PointCloud2ConstPtr tCloud;

//subscriber callback

 void enableDetection_cb(const std_msgs::String::ConstPtr& msg)
 {
     ROS_INFO("I Heard: [%s]", msg->data.c_str());
     std::string temp;
     temp = msg->data.c_str();

     if(temp.compare("Start") == 0)
     {
         startDetection = true;
     }
     if(temp.compare("Stop") == 0)
     {
         startDetection = false;
     }

     std::cout << "detection: " << startDetection << std::endl;

 }

 // cloud point callback

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    if(isFirstPC)
    {
        PointCloud cloud_in;
        pcl::fromROSMsg(*cloud,cloud_in);
        viewer.removeAllPointClouds();
        viewer.addPointCloud(cloud_in.makeShared(),"myCloud",0);
        ROS_INFO("Got First PointCloud!");
        isFirstPC = false;

    }
    if(pointsSelected)
    {
        gotPointCloud = true;
        pcl::fromROSMsg(*cloud,testCloud);
        viewer.removeAllPointClouds();
        viewer.removeAllPointClouds();
        //tCloud = *cloud;
        gotPointCloud = true;

    }
}

// selected points structure to hold point data and viewer pointer
struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloud::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

// point selection callback
void pointSelect_cb(const pcl::visualization::PointPickingEvent& event, void *args)
{
    struct callback_args* data = (struct callback_args *)args;

    if (event.getPointIndex() == -1)
        return;

    PointT current_point;
    event.getPoint(current_point.x,current_point.y,current_point.z);
    data->clicked_points_3d->points.push_back(current_point);

    // Add points to viewer


    pcl::visualization::PointCloudColorHandlerCustom<PointT> redPoint(data->clicked_points_3d,0,255,0);
    viewer.removePointCloud("clicked_points");
    viewer.addPointCloud(data->clicked_points_3d,redPoint,"clicked_points");

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,10,"clicked_points");

    std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

std::string get_selfpath() {
    char buff[PATH_MAX];
    char* path_end;

    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
    if (len != -1) {
      buff[len] = '\0';
      path_end = strrchr(buff,'/');
      ++path_end;
      buff[path_end-buff] = '\0';

      return std::string(buff);
    } else {
      /* handle error condition */
    }
}

// main function
int main(int argc, char **argv)
{
    isFirstPC = true;
    pointsSelected = false;
    gotPointCloud = false;
    startDetection = false;

    // load SVM file
    std::string selfpath = get_selfpath();
    std::cout << selfpath << std::endl;

    std::string svm_filename = "trainedLinearSVMForPeopleDetectionWithHOG.yaml";
    svm_filename = selfpath + svm_filename;

    // Algorithm parameters:
    float min_confidence = -1.5;
    float min_height = 1.3;
    float max_height = 2.3;
    float voxel_size = 0.06;
    Eigen::Matrix3f rgb_intrinsics_matrix;
    rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

    // Initialize ROS
    ros::init(argc,argv,"Human_Detector");
    ros::NodeHandle nh;
    ros::Rate loop_rate = 20;

    // set viewer camera location
    viewer.setCameraPosition(0,0,-2,0,-1,0,0);

    // Subscribe to point cloud
    sub = nh.subscribe("/camera/depth_registered/points",10,cloud_cb);
    // Subscribe to detector state
    enableDetection = nh.subscribe("/detector/state",1,enableDetection_cb);

    // Publisher initialize
    detectionOutcome = nh.advertise<std_msgs::String>("/detector/outcome",1);

    // Add point picking callback to viewer:
    struct callback_args cb_args;
    PointCloud::Ptr clicked_points_3d (new PointCloud);
    cb_args.clicked_points_3d = clicked_points_3d;
    cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
    viewer.registerPointPickingCallback (pointSelect_cb, (void*)&cb_args);

    int i = 0;
    while(!pointsSelected)
    {
        i++;
        ros::spinOnce();
        loop_rate.sleep();
        viewer.spinOnce();
        loop_rate.sleep();
        if(clicked_points_3d->points.size() == 3)
            pointsSelected = true;

    }
    // ground plane estimation
    Eigen::VectorXf ground_coeffs;
    ground_coeffs.resize(4);
    std::vector<int> clicked_points_indices;
    for(unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
            clicked_points_indices.push_back(i);
    pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
    model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
    // print plane coeffs
    std::cout << " Ground Plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " <<ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

    // reset cloudpoint from viewer
    viewer.removeAllPointClouds();
    viewer.removeAllShapes();

    // Create classifier for people detection
    pcl::people::PersonClassifier<pcl::RGB> person_classifier;
    person_classifier.loadSVMFromFile(svm_filename);

    // people detection app init
    pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;
    people_detector.setVoxelSize(voxel_size);
    people_detector.setIntrinsics(rgb_intrinsics_matrix);
    people_detector.setClassifier(person_classifier);
    people_detector.setHeightLimits(min_height,max_height);

    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
        if(gotPointCloud & startDetection)
        {
            PointCloud::Ptr aCloud (new PointCloud);
            aCloud = testCloud.makeShared();
            // start detection
            std::vector<pcl::people::PersonCluster<PointT> > clusters;
            people_detector.setInputCloud(aCloud);
            people_detector.setGround(ground_coeffs);
            people_detector.compute(clusters);

            ground_coeffs = people_detector.getGround();

            // draw cloud
            viewer.removeAllPointClouds();
            viewer.removeAllShapes();
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb (testCloud.makeShared());
            viewer.addPointCloud<PointT> (testCloud.makeShared(),rgb,"input");
            unsigned int k = 0;

            for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
            {
                if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
                    {
                          // draw theoretical person bounding box in the PCL viewer:
                          it->drawTBoundingBox(viewer, k);
                          k++;
                    }
            }
            if(k >= 0)
            {
                std::string out = static_cast<std::ostringstream*>( &(std::ostrstream() << k))->str();
                std_msgs::String msg;
                msg.data = out;
                detectionOutcome.publish(msg);
            }

            viewer.spinOnce();
        }

        gotPointCloud = false;
    }

}

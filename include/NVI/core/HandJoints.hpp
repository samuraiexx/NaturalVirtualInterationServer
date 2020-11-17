#include <Eigen/Geometry>
#include <opencv2/highgui.hpp>

template <typename T>
struct EulerAngle {
  EulerAngle(T forwardAngle = T(0), T sideAngle = T(0), T twistAngle = T(0)) :
    forwardAngle(forwardAngle), sideAngle(sideAngle), twistAngle(twistAngle) {}

  T forwardAngle = T(0);
  T sideAngle = T(0);
  T twistAngle = T(0);
};

template <typename T, typename V = double>
class HandJoints {
private:
  class Joint {
  public:
    Eigen::Matrix<T, 3, 1> offset;
    Eigen::Matrix<T, 3, 1> forwardAxis;
    Eigen::Matrix<T, 3, 1> sideAxis;
    Eigen::Matrix<T, 3, 1> twistAxis;

    Eigen::Quaternion<T> HandJoints<T, V>::Joint::rotateQuaternion(const Eigen::Quaternion<T> q, const EulerAngle<T>& angles);
    Joint();
  };

  static int GetJointParent(int joint);

  const V* const bonesSize;

  std::vector<std::vector<int>> children = {
    { 1, 5, 9, 13, 17 },
    { 2 }, { 3 }, { 4 }, {},
    { 6 }, { 7 }, { 8 }, {},
    { 10 }, { 11 }, { 12 }, {},
    { 14 }, { 15 }, { 16 }, {},
    { 18 }, { 19 }, { 20 }, {},
  };

  Joint joints[26];

  void forwardKinematicsDfs(
    int u,
    Eigen::Matrix<T, 3, 1> position,
    Eigen::Quaternion<T> rotation,
    const std::vector<EulerAngle<T>> &angles,
    std::vector<Eigen::Matrix<T, 3, 1>> &positions
  );

public:
  static void GetBonesSizeFromHandPoints(const std::vector<std::pair<cv::Point2d, double>> &stretchedHandPoints, double* bonesSize);
  static std::vector<EulerAngle<T>> GetEulerAnglesFromAnglesArray(const T* const angles);

  enum Joints {
    PULSE, THUMB_1, THUMB_2, THUMB_3, THUMB_4,
    INDEX_1, INDEX_2, INDEX_3, INDEX_4,
    MIDDLE_1, MIDDLE_2, MIDDLE_3, MIDDLE_4,
    RING_1, RING_2, RING_3, RING_4,
    LITTLE_1, LITTLE_2, LITTLE_3, LITTLE_4,
  };

  HandJoints(const V* const bonesSize);

  std::vector<Eigen::Matrix<T, 3, 1>> forwardKinematics(Eigen::Matrix<T, 3, 1> offset, const std::vector<EulerAngle<T>> &angles, const Eigen::Quaternion<T> &rotation);
};

template <typename T, typename V>
HandJoints<T, V>::HandJoints(const V* const bonesSize): bonesSize(bonesSize) {
  joints[1].offset << T(0.8989045451), T(-0.120882486), T(0.4211389834);
  joints[1].forwardAxis << T(0.4441698194), T(-0.2489050329), T(-0.8606739044);
  joints[1].sideAxis << T(-0.6401856542), T(-0.7602279186), T(-0.1105258316);
  joints[2].offset << T(0.6268027791), T(-0.6000759389), T(0.4970182528);
  joints[2].forwardAxis << T(-0.07430389524), T(-0.9245615005), T(-0.3737181425);
  joints[3].offset << T(0.5366801652), T(-0.3529426203), T(0.7664241039);
  joints[3].forwardAxis << T(-0.07085758448), T(-0.916480422), T(-0.3937550485);
  joints[4].offset << T(0.4737841707), T(-0.378308135), T(0.7952430538);

  joints[5].offset << T(0.3973699817), T(0.02934252055), T(0.9171892466);
  joints[5].forwardAxis << T(0.9269387722), T(-0.2924233675), T(-0.2351025343);
  joints[5].sideAxis << T(-0.2993878722), T(-0.9541106224), T(0.006337214261);
  joints[6].offset << T(0.2261557674), T(-0.06453067417), T(0.9719513162);
  joints[6].forwardAxis << T(0.9751105309), T(-0.09698209912), T(-0.1993838698);
  joints[7].offset << T(0.1692884395), T(-0.2552834754), T(0.9519305497);
  joints[7].forwardAxis << T(0.9390841126), T(-0.2454349846), T(-0.2405881733);
  joints[8].offset << T(0.1664149412), T(-0.2877388944), T(0.9431396482);

  joints[9].offset << T(0.08488206394), T(0.01530243113), T(0.996273492);
  joints[9].forwardAxis << T(0.9654377699), T(-0.2566545308), T(0.04536901787);
  joints[9].sideAxis << T(-0.2555440068), T(-0.9663660526), T(-0.02888046764);
  joints[10].offset << T(-0.05125549932), T(-0.01628844883), T(0.9985527328);
  joints[10].forwardAxis << T(0.9269410372), T(-0.3661398292), T(-0.08198827505);
  joints[11].offset << T(0.004476923166), T(-0.207669041), T(0.9781889013);
  joints[11].forwardAxis << T(0.9865300655), T(-0.163059175), T(-0.01304820739);
  joints[12].offset << T(-0.02122667917), T(-0.2067478551), T(0.9781639701);

  joints[13].offset << T(-0.187237788), T(0.03319196933), T(0.981753688);
  joints[13].forwardAxis << T(0.9344277382), T(-0.2928164899), T(0.2027393281);
  joints[13].sideAxis << T(-0.2848785222), T(-0.9561514854), T(-0.06796174496);
  joints[14].offset << T(-0.2137640341), T(-0.005778309047), T(0.9768682352);
  joints[14].forwardAxis << T(0.9818504453), T(-0.1556621641), T(0.1083459705);
  joints[15].offset << T(-0.1404158294), T(-0.2126081396), T(0.9669959534);
  joints[15].forwardAxis << T(0.9485346675), T(-0.2886407375), T(0.1302621663);
  joints[16].offset << T(-0.1946356855), T(-0.2068460125), T(0.9588178539);

  joints[17].offset << T(-0.4611206578), T(0.001345135421), T(0.8873364241);
  joints[17].forwardAxis << T(0.9627277851), T(-0.03340172768), T(0.2684019208);
  joints[17].sideAxis << T(0.01648308709), T(-0.9832555652), T(-0.1814859509);
  joints[18].offset << T(-0.2699706701), T(-0.1791500367),	T(0.9460555489);
  joints[18].forwardAxis << T(0.962085247), T(-0.1795446724), T(0.2053188086);
  joints[19].offset << T(-0.2534169667), T(-0.3101145421), T(0.9163017035);
  joints[19].forwardAxis << T(0.9082300067), T(-0.3974810243), T(0.1308711469);
  joints[20].offset << T(-0.2767889298), T(-0.336108193), T(0.9002328426);
}

template <typename T, typename V>
void HandJoints<T, V>::GetBonesSizeFromHandPoints(
  const std::vector<std::pair<cv::Point2d, double>> &stretchedHandPoints,
  double* bonesSize
) {
  bonesSize[0] = 0;
  for (int i = 1; i < 21; i++) {
    Point fromParent = stretchedHandPoints[i].first - stretchedHandPoints[GetJointParent(i)].first;
    T module = sqrt(fromParent.dot(fromParent));

    bonesSize[i] = module;
  }
}

template <typename T, typename V>
std::vector<EulerAngle<T>> HandJoints<T, V>::GetEulerAnglesFromAnglesArray(const T* const angles) {
  return {
    {}, // The pulse rotation is represented by a quaternion sent as the initial rotation
    {angles[0], angles[1]},
    {angles[2]},
    {angles[3]},
    {},
    {angles[4], angles[5]},
    {angles[6]},
    {angles[7]},
    {},
    {angles[8], angles[9]},
    {angles[10]},
    {angles[11]},
    {},
    {angles[12], angles[13]},
    {angles[14]},
    {angles[15]},
    {},
    {angles[16], angles[17]},
    {angles[18]},
    {angles[19]},
    {}
  };
}

template <typename T, typename V>
void HandJoints<T, V>::forwardKinematicsDfs(
  int u,
  Eigen::Matrix<T, 3, 1> position,
  Eigen::Quaternion<T> rotation,
  const std::vector<EulerAngle<T>>& angles,
  std::vector<Eigen::Matrix<T, 3, 1>>& positions
) {
  positions[u] = position + rotation*joints[u].offset*T(bonesSize[u]);
  Eigen::Quaternion<T> newRotation = joints[u].rotateQuaternion(rotation, angles[u]);

  for (int child : children[u]) {
    forwardKinematicsDfs(child, positions[u], newRotation, angles, positions);
  }
}

template <typename T, typename V>
HandJoints<T, V>::Joint::Joint() {
    offset << T(0), T(0), T(0);
    forwardAxis << T(1), T(0), T(0);
    sideAxis << T(0), T(1), T(0);
    twistAxis << T(0), T(0), T(1);
}

template <typename T, typename V>
std::vector<Eigen::Matrix<T, 3, 1>> HandJoints<T, V>::forwardKinematics(Eigen::Matrix<T, 3, 1> offset, const std::vector<EulerAngle<T>> &angles, const Eigen::Quaternion<T> &rotation) {
  std::vector<Eigen::Matrix<T, 3, 1>> positions(21);
  forwardKinematicsDfs(0, offset, rotation, angles, positions);

  return positions;
}

template <typename T, typename V>
int HandJoints<T, V>::GetJointParent(int joint) {
  switch (joint) {
  case 0:
  case 1:
  case 5:
  case 9:
  case 13:
  case 17:
    return 0;
  default:
    return joint - 1;
  }
}

template <typename T, typename V>
Eigen::Quaternion<T> HandJoints<T, V>::Joint::rotateQuaternion(const Eigen::Quaternion<T> q, const EulerAngle<T> &angles) {
  return q 
    * Eigen::AngleAxis<T>(-angles.forwardAngle, forwardAxis) 
    * Eigen::AngleAxis<T>(-angles.sideAngle, sideAxis) 
    * Eigen::AngleAxis<T>(-angles.twistAngle, twistAxis);
}


#include <iostream>
#include <chrono>
#include <cmath>
#include <atomic>
#include <thread>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
//basic //ros
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/client.h>
#include <sensor_msgs/Image.h>
#include "rc_msgs/detection.h"
#include "rc_msgs/results.h"
#include "rc_msgs/point.h"
#include "std_msgs/Bool.h"
#include "rc_msgs/stepConfig.h"
//ros   //msgs
#include <sensor_msgs/Image.h>
#include <rc_msgs/stepConfig.h>
#include "rc_msgs/detection.h"
#include "rc_msgs/results.h"
#include "rc_msgs/point.h"
#include "std_msgs/Bool.h"
//msgs  //yolov5
#include "yolo/cuda_utils.h"
#include "yolo/logging.h"
#include "yolo/utils.h"
#include "yolo/calibrator.h"
#include "yolo/preprocess.h"
//yolov5//yolov7
#include "yolo/yolo.hpp"

using namespace std;

// deleted
// deleted

//new
shared_ptr<Yolo::Infer> engine;
const auto using_mode = Yolo::Mode::FP16;   // 注意：修改此选项后必须删除原trtmodel才会生效
const int DEVICE = 0;
const int BATCH_SIZE = 1;
const float CONF_THRESH = .5;
const float NMS_THRESH = .4;
#define MAX_IMAGE_INPUT_SIZE_THRESH (3000 * 3000) // ensure it exceed the maximum size in the input images !

static const char *cocolabels[] = {
        "Tissues", "Toilet Rolls", "Soap", "Bottled Hand Sanitizer",
        "Porridge", "Chewing Gum", "Instant Noodles", "Potato Chips",
        "Canned Drinks", "Bottled Drinks", "Boxed Milk", "Bottled Water",
        "Apple", "Pear", "Banana", "Kiwi"
};

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        default:
            r = 1;
            g = 1;
            b = 1;
            break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
    float h_plane = (float) ((((unsigned int) id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = (float) ((((unsigned int) id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

inline long getModifyTime(char filePath[]) {
    struct stat buf{};
    auto fp = fopen(filePath, "r");
    int fd = fileno(fp);
    fstat(fd, &buf);
    fclose(fp);
    return buf.st_mtime;
}

inline long getNowTime() {
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec;
}

cv::Mat inference(const cv::Mat& image) {
    assert(engine != nullptr);

    shared_future<Yolo::BoxArray> box;
    for (int i = 0; i < 10; ++i)
        box = engine->commit(image);

    auto bbox = box.get();
    for (auto &obj: bbox) {
        uint8_t b, g, r;
        tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                      cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    return image;
}
//new

std::atomic<bool> beatRun(true);
int step = 0;
bool isIdentify = false;
ros::Publisher resPub;
ros::Publisher beatPub;
float *buffers[2];
uint8_t *img_host = nullptr;
uint8_t *img_device = nullptr;
//IExecutionContext *context;
cudaStream_t stream;
//float prob[BATCH_SIZE * OUTPUT_SIZE];
//int inputIndex;


void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    if (!isIdentify) {
        return;
    }

    rc_msgs::results Result;
    rc_msgs::detection tmp;
    cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    Result.step = step;
    assert(engine != nullptr);

    shared_future<Yolo::BoxArray> box;
    for (int i = 0; i < 10; ++i)
        box = engine->commit(img);

    auto bbox = box.get();
    for (auto &obj: bbox) {
        rc_msgs::detection tmp;
        uint8_t b, g, r;
        tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);
        tmp.x1 = obj.left;
        tmp.y1 = obj.top;
        tmp.x2 = obj.right;
        tmp.y2 = obj.bottom;
        auto name = cocolabels[obj.class_label];
        tmp.label = obj.class_label;
        tmp.score = obj.confidence;
        std::vector<rc_msgs::point> ps(4);
        ps[0].x = obj.left;
        ps[0].y = obj.top;
        ps[1].x = obj.right;
        ps[1].y = obj.top;
        ps[2].x = obj.right;
        ps[2].y = obj.bottom;
        ps[3].x = obj.left;
        ps[3].y = obj.bottom;
        tmp.contours = ps;
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(img, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                      cv::Scalar(b, g, r), -1);
        cv::putText(img, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        Result.results.emplace_back(tmp);
    }
    Result.color = *(cv_bridge::CvImage(std_msgs::Header(),"bgr8", img).toImageMsg());
    resPub.publish(Result);
//copy to memory and interface for v5
/*
    float *buffer_idx = (float *) buffers[inputIndex];
    size_t size_image = img.cols * img.rows * 3;
    size_t size_image_dst = INPUT_H * INPUT_W * 3;
    //copy data to pinned memory
    memcpy(img_host, img.data, size_image);
    //copy data to DEVICE memory
    CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
    buffer_idx += size_image_dst;

    // Run inference
    doInference(*context, stream, (void **) buffers, prob, BATCH_SIZE);
    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    auto &res = batch_res[0];
    nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
    for (auto &re: res) {
        cv::Rect r = get_rect(img, re.bbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string((int) re.class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        rc_msgs::detection tmp;
        tmp.x1 = r.x;
        tmp.y1 = r.y;
        tmp.x2 = r.x + r.width;
        tmp.y2 = r.y + r.height;
        tmp.label = (int) re.class_id;
        tmp.score = re.conf;
        std::vector<rc_msgs::point> ps(4);
        ps[0].x = r.x;
        ps[0].y = r.y;
        ps[1].x = r.x + r.width;
        ps[1].y = r.y;
        ps[2].x = r.x + r.width;
        ps[2].y = r.y + r.height;
        ps[3].x = r.x;
        ps[3].y = r.y + r.height;
        tmp.contours = ps;
        Result.results.push_back(tmp);
    }
*/
//copy to memory and interface for v5
}

void callback(const rc_msgs::stepConfig &config) {
    step = config.step;
}

void identifyCallback(const std_msgs::Bool::ConstPtr &msg) {
    isIdentify = msg->data;
}

void beatSend() {
    std::chrono::milliseconds duration(500);
    while (beatRun) {
        std_msgs::Bool beatMsg;
        beatMsg.data = true;
        beatPub.publish(beatMsg);
        std::this_thread::sleep_for(duration);
    }
}

//init for v7
shared_ptr<Yolo::Infer> init() {
    Yolo::set_device(DEVICE);

    if (access(ONNX_PATH, R_OK) != 0) {
        cerr << "ONNX File " << ONNX_PATH << " not exist!\n";
        exit(-1);
    }
    if (access(MODEL_PATH, R_OK) == 0) {
        long onnx_modify_time = getModifyTime(ONNX_PATH);
        long model_modify_time = getModifyTime(MODEL_PATH);
        long now_time = getNowTime();

        if (onnx_modify_time > now_time) {
            cerr << "ONNX File Modify Time is later than Now, Maybe Time Error!\n";
        }
        if (model_modify_time > now_time) {
            cerr << "Model File Modify Time is later than Now, Maybe Time Error!\n";
        }

        if (onnx_modify_time > model_modify_time) {
            cout << "Detected model file is older than onnx file, will rebuild engine!\n";
            Yolo::compile(
                    using_mode, Yolo::Type::V7,
                    BATCH_SIZE,
                    ONNX_PATH,
                    MODEL_PATH,
                    1 << 30,
                    "inference"
            );
        }
    } else {
        cout << "Model File " << MODEL_PATH << " not exist, will rebuild engine\n";
        Yolo::compile(
                using_mode, Yolo::Type::V7,
                BATCH_SIZE,
                ONNX_PATH,
                MODEL_PATH,
                1 << 30,
                "inference"
        );
    }

    return Yolo::create_infer(MODEL_PATH, Yolo::Type::V7, DEVICE, CONF_THRESH, NMS_THRESH);
}
//init for v7

int main(int argc, char **argv) {

//init for v5
/*
    cudaSetDevice(DEVICE);

    std::string engine_name(ENGINE_PATH);
    engine_name += "/engines/test.engine";

    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    std::cout << "start build engine\n";
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on DEVICE
    CUDA_CHECK(cudaMalloc((void **) &buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // prepare input data cache in pinned memory
    CUDA_CHECK(cudaMallocHost((void **) &img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in DEVICE memory
    CUDA_CHECK(cudaMalloc((void **) &img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    std::cout << "build ok\n";
*/
//init for v5
    engine = init();
    ros::init(argc, argv, "yolo_identify");
    ros::NodeHandle n;

    ros::Subscriber imageSub = n.subscribe("/raw_img", 1, &imageCallback);
    ros::Subscriber isIdentifySub = n.subscribe("/isIdentify", 1, &identifyCallback);
    resPub = n.advertise<rc_msgs::results>("/rcnn_results", 20);
    beatPub = n.advertise<std_msgs::Bool>("/nn_beat", 5);
    dynamic_reconfigure::Client<rc_msgs::stepConfig> client("/scheduler");

    client.setConfigurationCallback(&callback);

    std::thread beatThread = std::thread(&beatSend);

    ros::Rate loop_rate(10);
    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
    beatRun = false;
    beatThread.join();
// Release stream and buffers
/*
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
*/
    return 0;
}



#include "yolov7trt/yolo.hpp"

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

// 导出YoloV7 ONNX文件请采用下列指令
// python export.py --weights <your pt name>.pt --grid --simplify
// 将其重命名为cmake文件中规定的名字，放置于engine目录即可
// 程序启动时会检测是否需要重新编译engine

shared_ptr<Yolo::Infer> engine;
const auto using_mode = Yolo::Mode::FP16;   // 注意：修改此选项后必须删除原trtmodel才会生效
const int device = 0;
const int batch_size = 1;
const float conf_thresh = .5;
const float nms_thresh = .4;

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
    float h_plane = (float) ((((unsigned int) id << 2) ^ 0x937151) % 100) / 100.0f;;
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

void inference(const cv::Mat &image) {
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

    cv::imwrite("/home/stevegao/Downloads/qq-files/1315587032/file_recv/yolov7trt/result.jpg", image);
}

shared_ptr<Yolo::Infer> init() {
    Yolo::set_device(device);

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
                    batch_size,
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
                batch_size,
                ONNX_PATH,
                MODEL_PATH,
                1 << 30,
                "inference"
        );
    }

    return Yolo::create_infer(MODEL_PATH, Yolo::Type::V7, device, conf_thresh, nms_thresh);
}

int main() {
    engine = init();

    auto test_image = cv::imread("/home/stevegao/Downloads/qq-files/1315587032/file_recv/yolov7trt/test.jpg");
    inference(test_image);

    engine.reset();
}
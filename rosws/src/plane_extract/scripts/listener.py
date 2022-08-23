# coding=utf-8
import rospy 
from calibrateResult.msg import calibrateResult 
from point.msg import point
def callback(msgg):  #msgg的名字随便定义,保持一致即可
	rospy.loginfo("I heared:%6.2f %6.2f\n",msgg.data[0].x,msgg.data[0].y) #数组的大小要与talker里面的一样
	

	
def listener(): 
	rospy.init_node('listener', anonymous= True)
	rospy.Subscriber("/calibrateResult", calibrateResult, callback) 
	rospy.spin() 
if __name__ == '__main__': 
	listener() 


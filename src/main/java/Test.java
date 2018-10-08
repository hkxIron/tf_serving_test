/*
 * Created by IntelliJ IDEA.
 * Author: hukexin
 * Date: 18-10-8
 * Time: 下午5:43
 * 参考blog: https://blog.csdn.net/luoyexuge/article/details/80330457
 */

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import java.util.Arrays;

public class Test {

    SavedModelBundle bundle = null;

    public static SavedModelBundle loadModel(String modelpath){
        System.out.println("begin to load model from path:"+modelpath);
        SavedModelBundle savedModelBundle=SavedModelBundle.load(modelpath,"serve"); // 这里的tags为python里add_meta_graph_and_variables的tags必须一致
        return savedModelBundle;
    }


    public void init() {
        String  classpath = this.getClass().getResource("/").getPath()+"model/1" ;
        bundle=loadModel(classpath);
    }

    public  double  getResult(float[][] arr){
        Tensor  tensor=Tensor.create(arr);
        Tensor<?>  result= bundle
                .session()
                .runner()
                .feed("x",tensor)
                .fetch("y")
                .run()
                .get(0);
        float[][] resultValues = (float[][])result.copyTo(new float[1][1]);
        result.close(); // session需要关闭
        return resultValues[0][0];

    }

    public static void main(String[] args){
        Test model = new Test();
        model.init();
        float[][] arr=new float[1][3];
        arr[0][0]=1f;
        arr[0][1]=0.5f;
        arr[0][2]=2.0f;
        double result = model.getResult(arr);
        System.out.println("predict result:"+ result);
        //System.out.println(Arrays.toString("他".getBytes()));
    }

}


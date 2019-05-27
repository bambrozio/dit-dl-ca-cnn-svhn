# Comments are texts extracted from assignment specification

# here I will import the main module from your code - you need to make sure your code imports without a problem
# As per the assignment specification, your main module must be called svhn.py
import svhn

def main():
    # I might start by calling on your code to do some processing based on the model that you already trained

    # JPG from internet
    result1 = svhn.test("img_samples/internet/246x0w.jpg") #123
    print(result1)

    # i might also test with a PNG

    # Images from full dataset, preprocessed
    result2 = svhn.test("img_samples/cropped/16.png") #14
    print(result2)

    result3 = svhn.test("img_samples/cropped/19.png") #60
    print(result3)

    # Image from full dataset, non-preprocessed
    result4 = svhn.test("img_samples/full_ds_img/19.png") #60
    print(result4)

    # Image from full dataset, non-preprocessed
    result5 = svhn.test("img_samples/full_ds_img/323.png")  # 2
    print(result5)

    ##############################
    ##############################

    # # I will also call to start training on your code from scratch. I might not always wait for training to complete
    # # but I will start the training and make sure it is progressing.
    # average_f1_scores = svhn.traintest()
    # print(average_f1_scores)


if __name__ == '__main__':
    main()

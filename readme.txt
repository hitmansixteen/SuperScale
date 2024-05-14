==========================Super Scale=============================

-----------------------How to use code----------------------------


RUNNING SERVER ONLY:

	*To only test the code go to app.py and run it.

	*Then open your browser and type 127.0.0.1:5000

	*Wait for server to run
	
	*Upload Image
	
	*Wait for image to comeback

	*Download Image


TRAINING THE MODEL:

	*To train the model place your images in "DIV2K_train_HR" folder

	*Run the code

	*Wait for 250 epochs to finish or you can just stop it at any epoch you want it saves progress after every epoch

	*Your model weights will be stored in "weights" folder

	*you can take out any tarfile to root folder and extract it and then use it for the running code on server
	 you will have to change the path or name of .pth file in code of app.py

	*g_loss_data will have generator loss after every batch and d_loss_data will have discriminator loss after every batch

TESTING THE MODEL:

	*To test the model place high resolution files in "testdata", place low resolution file in "testlrdata"

	*Make sure to the names of paired images are same

	*run the code

	*graphs folder will have graph of avg metric in each subfolder over the entire testing phase of 250 weights

	*change value of epoch if you have less weights in weights folder

	*mse.txt psnr.txt ssim.txt will have value against every image in every epoch

	*Testfolder/div2k will have generated test images
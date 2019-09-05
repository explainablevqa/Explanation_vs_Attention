import scipy.misc
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')

fig = plt.figure(figsize = (7, 6))# this is no of row and col (4, 11)  and figure size should be reverse of this , i.e (33, 12) ratio
fig.subplots_adjust(left=0, right=1.0, bottom=0.15, top=0.95, hspace=0.6, wspace=0.0)

img1Idx = 5
img2Idx = 9
nEpoch = 127

if len(sys.argv) > 2:
	img1Idx = sys.argv[1]
	img2Idx = sys.argv[2]

if len(sys.argv) > 3:
    nEpoch = sys.argv[3]

ques = ["Does it almost seem that\nthe fence comes out\nof hiding as it nears\nthe woman's feet?\nAns : yes",
		"Would it be a good idea\nto put goldfish inside the\nopen part of this fixture?\nAns : no",
		"If the girls are not wearing\nwhite socks what color socks\nare they wearing?\nAns : blue",
		"Why would the snowmobiler\nbe riding up the mountain\nfor the skier?\nAns : rescue",
		"How symmetrical are the white\nbricks an either side of\nthe building?\nAns: very",
		"What mode of transportation\nis in the center of the photo?\nAns : train",
		"What does the person in\nthis picture have on is face?\nAns : glass",
		"Where should I go to\nget to the book sale?\nAns : left",
		"Does this flask have enough\nwine to fill the glass?\nAns : yes",
		"Is the dog only have\none of his eyes showing?\nAns : yes",
		"What do you call the wooden\nitems behind the bear?\nAns : pallets",
		"Does this look like a young\nor an old elephant?\nAns : young",
		"Do you think an adult\nis monitorig this child?\nAns : yes",
		"Is it a left or right hand\nholding the object?\nAns : left"]


imgName = ['../pytorch-vqa_att_new/Results/RawImages/ep150_cnt0_' + str(img1Idx) + 'raw.png', 'Results/GradcamImages/ep' + str(nEpoch) + '_cnt0_' + str(img1Idx) + 'gradcam.png', 'Results/AttImages/ep' + str(nEpoch) + '_cnt0_' + str(img1Idx) + 'att.png',
		   '../pytorch-vqa_att_new/Results/RawImages/ep150_cnt0_' + str(img2Idx) + 'raw.png', 'Results/GradcamImages/ep' + str(nEpoch) + '_cnt0_' + str(img2Idx) + 'gradcam.png', 'Results/AttImages/ep' + str(nEpoch) + '_cnt0_' + str(img2Idx) + 'att.png']
xLabel = [ques[int(img1Idx)], '\nGradcam Visualization', '\nAttention Visualization',
		  ques[int(img2Idx)], '\nGradcam Visualization', '\nAttention Visualization']

for i in range(6):
	plt.subplot(2, 3, i+1)
	plt.xticks(())
	plt.yticks(())
	orig_img = scipy.misc.imread(imgName[i])
	plt.imshow(scipy.misc.imresize(orig_img, (300, 300)))
	plt.xlabel(xLabel[i])

plt.savefig('vqa_att_grad_' + str(img1Idx) + '_' + str(img2Idx) + '.png')

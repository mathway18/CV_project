# Project Oral Presentation Notes

## 1. Recommended Presentation Structure

Target length: about 8 minutes  
Recommended slides: 7 to 8 pages

### Slide 1. Title
**Video Object Removal and Background Restoration**  
AIAA 3201 Introduction to Computer Vision  
Group members: `[Your Names]`

Suggested speaking:

> Hello everyone. Our project is video object removal and background restoration. The goal is to automatically detect dynamic objects in a video, remove them, and recover a clean background. At this stage, we have completed Part 1 of the project, which is the hand-crafted baseline pipeline.

---

### Slide 2. Task Definition
Title: **Problem and Goal**

Suggested slide bullets:

- Input: video containing dynamic objects such as pedestrians, bicycles, or animals
- Output: video with dynamic objects removed and background restored
- Main challenges:
- Accurate dynamic object segmentation
- Distinguishing moving objects from static objects
- Filling missing regions in a temporally consistent way

Suggested speaking:

> Our input is a normal video that contains moving objects. The output should be a clean video where the dynamic objects are removed. This task is challenging because we need reliable masks, we need to distinguish moving objects from static objects, and we also need to restore the background without obvious artifacts.

---

### Slide 3. Current Progress
Title: **Completed Part 1 Baseline**

Suggested slide bullets:

- We implemented the full Part 1 pipeline required by the project PDF
- Pipeline:
- YOLOv8-Seg for object detection and segmentation
- Lucas-Kanade optical flow for dynamic judgment
- Dilation for mask refinement
- Temporal propagation plus OpenCV inpainting for background restoration

Suggested speaking:

> We have completed the first part of the project. Our current system first detects object regions using YOLOv8-Seg, then uses optical flow to determine whether the object is actually moving. After that, we enlarge the mask slightly using dilation, and finally restore the removed area using temporal borrowing and OpenCV inpainting.

---

### Slide 4. Method Overview
Title: **Baseline Pipeline**

Suggested slide content:

`Input Video -> Segmentation -> Motion Filtering -> Mask Dilation -> Temporal Fill -> Spatial Inpainting -> Output Video`

Suggested speaking:

> This slide shows our full baseline pipeline. First, we process each frame and extract candidate object masks. Then we use motion information to filter out static objects. Next, we refine the mask boundary with dilation. Finally, we restore the missing background. We first try to borrow clean pixels from previous frames, and if some regions are still missing, we apply spatial inpainting.

---

### Slide 5. Method Details
Title: **Dynamic Mask Extraction**

Suggested slide bullets:

- YOLOv8-Seg detects target classes such as person, bicycle, and cat
- Optical flow is computed inside each detected region
- If average motion is above a threshold, the object is treated as dynamic
- Dilation is applied to improve coverage near motion boundaries

Suggested speaking:

> In the mask extraction stage, we use YOLOv8-Seg to get candidate masks for several dynamic object categories. However, not every detected object should be removed. So we compute optical flow inside the detected region and use the motion magnitude to decide whether the object is dynamic. We also use dilation because the original mask may be too tight and may miss motion blur or boundary pixels.

---

### Slide 6. Method Details
Title: **Background Restoration**

Suggested slide bullets:

- First try temporal background propagation
- Use clean pixels from previous frames at the same location
- Use OpenCV inpainting as fallback for remaining missing regions
- This is simple and works reasonably well for static backgrounds

Suggested speaking:

> For restoration, we first use a temporal strategy. If a region is masked in the current frame but was visible in a previous frame, we directly borrow the background from the previous frame. This uses the temporal redundancy of the video. For any remaining holes, we use OpenCV inpainting. This method is lightweight and easy to implement, but it is still limited in complex scenes.

---

### Slide 7. Experiments and Results
Title: **Experimental Results**

Suggested slide bullets:

- Tested on short videos with moving objects
- Saved:
- original frames
- mask visualization
- overlay visualization
- restored video
- Observations:
- Works on simple scenes with relatively static backgrounds
- Fails on large occlusion or complex textures
- Some blur remains after inpainting

Suggested speaking:

> For experiments, we tested our method on videos containing moving objects. We generated several types of outputs, including masks, overlay videos, and restored videos. Qualitatively, the method works reasonably well when the background is simple and stable. However, when the removed object covers a large region or when the background contains complex textures, the result may become blurry or inconsistent.

---

### Slide 8. Limitations and Next Plan
Title: **Next Step: Part 2**

Suggested slide bullets:

- Current limitations:
- mask quality is not always precise
- restoration quality degrades for large holes
- temporal consistency is still limited
- Next plan:
- use SAM 2 for higher-quality dynamic masks
- use ProPainter for stronger video inpainting
- evaluate on mandatory datasets and add quantitative metrics

Suggested speaking:

> Our current baseline already completes the first part of the project, but it still has several limitations. The mask boundary is not always accurate, and the restoration result degrades when the missing area is large. Our next step is to build the Part 2 AI-driven pipeline using SAM 2 for better mask extraction and ProPainter for stronger video inpainting. We also plan to evaluate on the required datasets and report quantitative metrics such as PSNR and SSIM.

---

## 2. One-Sentence Closing

> In summary, we have successfully completed the Part 1 baseline pipeline, observed both its strengths and weaknesses, and we are now moving toward a stronger AI-driven solution for Part 2.

## 3. Recording Advice

Recommended recording structure:

1. Open the PPT in full-screen mode.
2. Talk through slides 1 to 8.
3. On the experiment slide, briefly play the generated side-by-side comparison video.
4. End on the next-step slide.

Keep the video around 7 minutes 30 seconds to 8 minutes.

## 4. What To Put On the Experiment Slide

Use one slide with:

- One pipeline figure
- Two rows of visual results
- Each row contains:
- original frame
- mask or overlay
- restored frame
- one short note under each example

If you do not have many experiments yet, showing 2 videos carefully is enough for this oral update.

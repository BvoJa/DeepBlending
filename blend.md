# Object Blending Through Deep Image Blending

This note explains how the input images go through Deep Image Blending for object blending in this codebase. The important inputs are:

- `source_img`: source image, the image containing the object to blend.
- `target_img`: target image, the image that receives the object.
- `mask_img`: object mask selected on the source image.
- `x_start`, `y_start`: target placement, stored as the center of the pasted source window.
- `ss`, `ts`: source and target resize sizes used before optimization.

The main execution path is:

```text
run.py
  -> load source/target/mask images
  -> make_canvas_mask(...)
  -> compute_gt_gradient(...)
  -> convert images and masks to torch tensors
  -> initialize input_img as the optimizable image
  -> LBFGS first pass closure
      -> composite input_img foreground over target_img background
      -> laplacian_filter_tensor(...)
      -> Vgg16/MeanShift/gram_matrix losses
      -> optimizer.step(...)
  -> save first_pass.png
  -> reload first_pass.png and target image
  -> LBFGS second pass closure
      -> Vgg16/MeanShift/gram_matrix refinement
      -> optimizer.step(...)
  -> save second_pass.png
```

## 1. Inputs Enter `run.py`

The object blending script receives the source image, mask image, target image, output directory, source size, target size, placement center, GPU id, optimization steps, and optional video flag through command-line arguments. Code: [run.py:18-30](run.py#L18-L30).

Those parsed arguments are copied into the first-pass input variables: `source_file`, `mask_file`, `target_file`, `gpu_id`, `num_steps`, `ss`, `ts`, `x_start`, and `y_start`. The first pass also sets the gradient, style, content, and total-variation weights. Code: [run.py:41-54](run.py#L41-L54).

## 2. Files Become Resized Image Arrays

The source and target files are opened as RGB images, then resized to `ss x ss` and `ts x ts`. The mask is opened as a grayscale image, resized to the source size, and binarized so every positive value becomes `1`. Code: [run.py:56-60](run.py#L56-L60).

At this point the pipeline has three NumPy arrays:

- `source_img`: RGB source object image at source resolution.
- `target_img`: RGB receiving image at target resolution.
- `mask_img`: binary source-space object mask at source resolution.

## 3. The Source Mask Is Placed on the Target Canvas

`run.py` calls `make_canvas_mask` to create a target-sized mask showing where the source object window should land in the target image. Code: [run.py:62-65](run.py#L62-L65).

Inside `make_canvas_mask`, the code creates a zero mask with the target image height and width, then inserts the source mask into the rectangle centered at `(x_start, y_start)`. Code: [utils.py:28-31](utils.py#L28-L31).

The canvas mask is then converted to a CUDA tensor and expanded from one channel to three channels so it can mask RGB image tensors. Code: [run.py:63-65](run.py#L63-L65), [utils.py:20-25](utils.py#L20-L25).

In short:

- `mask_img`: object mask in source-image coordinates.
- `canvas_mask`: same object placement, but in target-image coordinates.

## 4. The Code Builds Ground-Truth Poisson Gradients

Before optimization, `run.py` computes `gt_gradient`, the gradient-domain target that the blended result should match. Code: [run.py:67-68](run.py#L67-L68).

`compute_gt_gradient` first converts the source and target arrays into CUDA tensors and applies `laplacian_filter_tensor` to both images. Code: [utils.py:52-66](utils.py#L52-L66).

`laplacian_filter_tensor` defines a 3x3 Laplacian kernel, applies it separately to the red, green, and blue channels, and returns one gradient tensor per channel. Code: [utils.py:33-49](utils.py#L33-L49).

Then `compute_gt_gradient` creates a target-sized canvas mask, keeps source-image gradients only inside the object mask, places those foreground gradients into the target canvas, keeps target-image gradients outside the object region, and adds foreground plus background gradients for each channel. Code: [utils.py:68-91](utils.py#L68-L91).

Finally, the three target gradient maps are converted back into tensors and returned as `gt_gradient`. Code: [utils.py:101-106](utils.py#L101-L106).

## 5. Pixels Become Optimizable Tensors

The source and target arrays are converted from `H x W x C` NumPy layout into batched `N x C x H x W` CUDA tensors. Code: [run.py:70-72](run.py#L70-L72).

The blend itself starts as random pixels with the same tensor shape as the target image:

```python
input_img = torch.randn(target_img.shape).to(gpu_id)
```

Code: [run.py:73](run.py#L73).

The source mask is also converted to a tensor and expanded to three channels so it can mask the RGB source object during the content loss. Code: [run.py:75-76](run.py#L75-L76), [utils.py:20-25](utils.py#L20-L25).

The optimizer is L-BFGS over `input_img.requires_grad_()`, so the algorithm directly updates pixel values rather than network weights. Code: [run.py:78-82](run.py#L78-L82).

The VGG feature model and mean-shift preprocessing are created once before optimization. Code: [run.py:87-89](run.py#L87-L89), [utils.py:111-142](utils.py#L111-L142), [utils.py:160-171](utils.py#L160-L171).

## 6. The First Pass Composites the Current Object Region

Each L-BFGS step runs the closure beginning at the first-pass loop. Code: [run.py:95-98](run.py#L95-L98).

Inside the closure, the current `input_img` is used only where `canvas_mask` is `1`, while the target image is kept outside the mask:

```python
blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)
```

Code: [run.py:99-102](run.py#L99-L102).

This is the pixel-space blend candidate used for the first-pass gradient, content, and total-variation losses. The source object is not copied directly into the target; instead, the masked region of `input_img` is optimized until it has source-object content and target-compatible boundaries.

## 7. The First Pass Pulls the Blend With Four Losses

The gradient loss applies the Laplacian filter to `blend_img`, compares every color-channel gradient against `gt_gradient`, averages the channel losses, and multiplies by `grad_weight`. Code: [run.py:103-111](run.py#L103-L111), [utils.py:33-49](utils.py#L33-L49).

The style loss extracts VGG features from the target image and from `input_img`, converts those features to Gram matrices, and compares the Gram matrices layer by layer. Code: [run.py:113-124](run.py#L113-L124), [utils.py:144-149](utils.py#L144-L149).

The content loss crops the blended object window from `blend_img`, masks the source and blended object tensors with `mask_img`, extracts VGG features, and compares their `relu2_2` activations. Code: [run.py:127-132](run.py#L127-L132).

The total-variation loss penalizes horizontal and vertical pixel differences in `blend_img`, which discourages noisy optimized pixels. Code: [run.py:134-137](run.py#L134-L137).

The first-pass objective is the sum of gradient, style, content, and TV losses. The closure clears old gradients and backpropagates through the current pixel tensor. Code: [run.py:139-142](run.py#L139-L142).

## 8. L-BFGS Updates the First-Pass Image

After the closure is defined, L-BFGS calls it through `optimizer.step(closure)`, repeatedly updating `input_img` for the configured number of optimization steps. Code: [run.py:95-96](run.py#L95-L96), [run.py:172](run.py#L172).

When the loop finishes, the optimized pixels are clamped to `[0, 255]`. The code composites the optimized foreground region over the target background one final time, converts the tensor back to a NumPy image, and writes `first_pass.png`. Code: [run.py:174-184](run.py#L174-L184).

The first-pass result is therefore:

- target pixels outside `canvas_mask`;
- optimized pixels inside `canvas_mask`;
- boundaries guided by the mixed source/target Laplacian gradients.

## 9. The First-Pass Result Enters the Second Pass

The second pass resets the style and content weights, fixes both source and target sizes to `512`, and reuses the requested number of optimization steps. Code: [run.py:190-193](run.py#L190-L193).

It reloads the saved `first_pass.png` and the original target image, converts both into batched CUDA tensors, and makes them contiguous. Code: [run.py:195-201](run.py#L195-L201).

The second optimizer is L-BFGS over `first_pass_img.requires_grad_()`, so this pass directly updates the whole first-pass image tensor. Code: [run.py:203-208](run.py#L203-L208).

## 10. The Second Pass Refines the Whole Image

During the second-pass closure, the code extracts VGG features from the target image and the current `first_pass_img`, computes Gram matrices, and optimizes a much stronger style loss against the target. Code: [run.py:210-225](run.py#L210-L225), [utils.py:144-149](utils.py#L144-L149).

The content loss is also computed from VGG `relu2_2` features. As written in `run.py`, both sides of that content comparison come from the current `first_pass_img`, so the second-pass loss is effectively driven by the style term in this script. Code: [run.py:227-232](run.py#L227-L232).

L-BFGS updates `first_pass_img` through the closure. After optimization, the image is clamped to `[0, 255]`, converted back to NumPy layout, and saved as `second_pass.png`. Code: [run.py:257-266](run.py#L257-L266).

## 11. Optional Reconstruction Video Follows the Same Composite

If `--save_video` is enabled, the code opens a video writer before optimization. Code: [run.py:91-93](run.py#L91-L93).

During the first pass, the saved video frame is built from the same masked foreground plus target background composite used by the blending loss. Code: [run.py:144-156](run.py#L144-L156).

During the second pass, the video path composites `first_pass_img` through the old `canvas_mask` and appends each frame. Code: [run.py:236-243](run.py#L236-L243).

After the final image is saved, the video writer is closed. Code: [run.py:268-270](run.py#L268-L270).

## Short Summary

The object is blended by direct pixel optimization, not by a learned generator or diffusion latent process. The code first resizes the source, target, and mask; places the source mask onto a target-sized canvas; builds a mixed Poisson-style gradient target from source gradients inside the object and target gradients outside it; then optimizes a target-sized pixel tensor with L-BFGS.

The first pass is the actual object blending pass:

1. **Canvas compositing** uses optimized pixels inside the placed mask and target pixels outside it. Code: [run.py:99-102](run.py#L99-L102).
2. **Poisson-style gradient loss** matches source-object gradients inside the mask and target-background gradients outside it. Code: [utils.py:68-91](utils.py#L68-L91), [run.py:103-111](run.py#L103-L111).
3. **VGG style and content losses** make the object region keep source content while taking texture/style cues from the target image. Code: [run.py:113-132](run.py#L113-L132), [utils.py:111-149](utils.py#L111-L149).
4. **Second-pass refinement** reloads `first_pass.png` and optimizes the whole image toward target-style VGG statistics before saving `second_pass.png`. Code: [run.py:195-208](run.py#L195-L208), [run.py:210-266](run.py#L210-L266).

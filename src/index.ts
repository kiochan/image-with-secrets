import cv from '@techstark/opencv-js';
import Jimp from 'jimp';

function shiftDft(mag: cv.Mat): void {
  const rect = new cv.Rect(0, 0, mag.cols & -2, mag.rows & -2);
  mag.roi(rect);

  const cx = mag.cols / 2;
  const cy = mag.rows / 2;

  const q0 = mag.roi(new cv.Rect(0, 0, cx, cy));
  const q1 = mag.roi(new cv.Rect(cx, 0, cx, cy));
  const q2 = mag.roi(new cv.Rect(0, cy, cx, cy));
  const q3 = mag.roi(new cv.Rect(cx, cy, cx, cy));

  const tmp = new cv.Mat();
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  tmp.delete();
  q0.delete();
  q1.delete();
  q2.delete();
  q3.delete();
}

function getBlueChannel(image: cv.Mat): cv.Mat {
  const channel = new cv.MatVector();
  cv.split(image, channel);
  return channel.get(0);
}

function getDftMat(padded: cv.Mat): cv.Mat {
  const planes = new cv.MatVector();
  planes.push_back(padded);
  const matZ = cv.Mat.zeros(padded.size(), cv.CV_32F);
  planes.push_back(matZ);
  const comImg = new cv.Mat();
  cv.merge(planes, comImg);
  cv.dft(comImg, comImg);
  matZ.delete();
  return comImg;
}

function addTextByMat(
  comImg: cv.Mat,
  text: string,
  point: cv.Point,
  fontSize: number
): void {
  cv.putText(
    comImg,
    text,
    point,
    cv.FONT_HERSHEY_DUPLEX,
    fontSize,
    cv.Scalar.all(0),
    2
  );
  cv.flip(comImg, comImg, -1);
  cv.putText(
    comImg,
    text,
    point,
    cv.FONT_HERSHEY_DUPLEX,
    fontSize,
    cv.Scalar.all(0),
    2
  );
  cv.flip(comImg, comImg, -1);
}

function writeTextIntoMat(mat: cv.Mat, text: string, fontSize: number): cv.Mat {
  const padded = getBlueChannel(mat);
  padded.convertTo(padded, cv.CV_32F);
  const comImg = getDftMat(padded);
  // add text
  const center = new cv.Point(padded.cols / 2, padded.rows / 2);
  addTextByMat(comImg, text, center, fontSize);
  const outer = new cv.Point(45, 45);
  addTextByMat(comImg, text, outer, fontSize);
  //back image
  const invDFT = new cv.Mat();
  cv.idft(comImg, invDFT, cv.DFT_SCALE | cv.DFT_REAL_OUTPUT, 0);
  const restoredImage = new cv.Mat();
  invDFT.convertTo(restoredImage, cv.CV_8U);
  const backPlanes = new cv.MatVector();
  cv.split(mat, backPlanes);
  backPlanes.set(0, restoredImage);
  const backImage = new cv.Mat();
  cv.merge(backPlanes, backImage);

  padded.delete();
  comImg.delete();
  invDFT.delete();
  restoredImage.delete();
  return backImage;
}

function getTextFormMat(backImage: cv.Mat): cv.Mat {
  const padded = getBlueChannel(backImage);
  padded.convertTo(padded, cv.CV_32F);
  const comImg = getDftMat(padded);
  const backPlanes = new cv.MatVector();
  // split the complex image in two backPlanes
  cv.split(comImg, backPlanes);
  const magnitude = new cv.Mat();
  // compute the magnitude
  cv.magnitude(backPlanes.get(0), backPlanes.get(1), magnitude);
  // move to a logarithmic scale
  const matOne = cv.Mat.ones(magnitude.size(), cv.CV_32F);
  cv.add(matOne, magnitude, magnitude);
  cv.log(magnitude, magnitude);
  shiftDft(magnitude);
  magnitude.convertTo(magnitude, cv.CV_8UC1);
  cv.normalize(magnitude, magnitude, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1);

  padded.delete();
  comImg.delete();
  matOne.delete();
  return magnitude;
}

function matToBuffer(mat: cv.Mat): Buffer {
  const img = new cv.Mat();
  const depth = mat.type() % 8;
  const scale = depth <= cv.CV_8S ? 1 : depth <= cv.CV_32S ? 1 / 256 : 255;
  const shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128 : 0;
  mat.convertTo(img, cv.CV_8U, scale, shift);
  const type = img.type();
  switch (type) {
    case cv.CV_8UC1:
      cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
      break;
    case cv.CV_8UC3:
      cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
      break;
    case cv.CV_8UC4:
      break;
    default:
      throw new Error(
        `Unexpected number of channel (expected 1, 3 or 4 but results of type is [${type}])`
      );
  }
  const imgData = Buffer.from(img.data);
  img.delete();
  return imgData;
}

export async function writeTextIntoImage(
  image: Buffer,
  watermarkText: string,
  fontSize: number = 16
): Promise<Buffer> {
  const jimpSrc: Jimp = await Jimp.read(image);
  const srcImg = cv.matFromImageData(jimpSrc.bitmap as unknown as cv.ImageData);
  if (srcImg.empty()) {
    throw new Error('read image failed');
  }
  const comImg = writeTextIntoMat(srcImg, watermarkText, fontSize);
  const imgRes = new Jimp({
    width: comImg.cols,
    height: comImg.rows,
    data: matToBuffer(comImg)
  });
  srcImg.delete();
  comImg.delete();
  return await imgRes.getBufferAsync(Jimp.MIME_PNG);
}

export async function getTextFormImage(buffer: Buffer): Promise<Buffer> {
  const jimpSrc = await Jimp.read(buffer);
  const comImg = cv.matFromImageData(jimpSrc.bitmap);
  const backImage = getTextFormMat(comImg);
  const imgRes = new Jimp({
    width: backImage.cols,
    height: backImage.rows,
    data: matToBuffer(backImage)
  });
  comImg.delete();
  backImage.delete();
  return await imgRes.getBufferAsync(Jimp.MIME_PNG);
}

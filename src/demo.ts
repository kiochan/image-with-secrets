import { getTextFormImage, writeTextIntoImage } from './';
import { readFile, writeFile } from 'fs/promises';

// top level await is evil!
async function main() {
  const secretText = 'lenna';

  const image = await readFile('../examples/lenna.png');

  const imageWithSecrets = await writeTextIntoImage(image, secretText);

  const secretTextImage = await getTextFormImage(imageWithSecrets);

  await writeFile('../tmp/lenna', secretTextImage);
}

main();

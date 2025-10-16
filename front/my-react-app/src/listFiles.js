const fs = require('fs');
const path = require('path');

const targetDirectory = path.join(__dirname, '..', '..', '..', 'storage');
const outputFile = path.join(__dirname, 'file-list.json');
const publicDir = path.join(__dirname, '..', 'public');
const publicStorage = path.join(publicDir, 'storage');

if (!fs.existsSync(targetDirectory)) {
  console.error(`Storage directory not found: ${targetDirectory}`);
  process.exit(1);
}

console.log(`Reading files from ${targetDirectory}`);
const files = fs.readdirSync(targetDirectory);
fs.writeFileSync(outputFile, JSON.stringify(files, null, 2));
console.log(`Successfully generated file list at ${outputFile}`);

function ensurePublicSymlink() {
  const relativeTarget = path.relative(publicDir, targetDirectory);

  try {
    if (fs.existsSync(publicStorage)) {
      const stats = fs.lstatSync(publicStorage);
      if (stats.isSymbolicLink()) {
        const currentTarget = fs.readlinkSync(publicStorage);
        if (currentTarget === relativeTarget) {
          console.log(`Symlink already exists at ${publicStorage}`);
          return;
        }

        console.log(
          `Updating symlink at ${publicStorage} to point to ${relativeTarget}`
        );
        fs.unlinkSync(publicStorage);
      } else {
        console.warn(
          `Cannot create symlink: ${publicStorage} exists and is not a symlink.`
        );
        return;
      }
    }

    fs.symlinkSync(relativeTarget, publicStorage, 'junction');
    console.log(`Created symlink ${publicStorage} -> ${relativeTarget}`);
  } catch (error) {
    console.warn(
      `Failed to ensure public storage symlink: ${(error && error.message) || error}`
    );
  }
}

ensurePublicSymlink();

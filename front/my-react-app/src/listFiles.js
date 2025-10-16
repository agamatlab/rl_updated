const fs = require('fs');
const path =  require('path');

const targetDirectory = path.join(__dirname, '..', '..','..','storage')
const outputFile = path.join(__dirname, 'file-list.json');

console.log(`Reading files from ${targetDirectory}`);
const files = fs.readdirSync(targetDirectory);

fs.writeFileSync(outputFile, JSON.stringify(files, null, 2));
console.log(`Successfully generated file list at ${outputFile}`);
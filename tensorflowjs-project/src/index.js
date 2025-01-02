// Este arquivo é o ponto de entrada da aplicação. Ele configura o ambiente TensorFlow.js, carrega o modelo e executa inferências com base nos dados de entrada.

import * as tf from '@tensorflow/tfjs';

// Carregar o modelo
async function loadModel() {
    const model = await tf.loadLayersModel('model/model.image.json');
    return model;
}

// Pré-processar a imagem de entrada
function preprocessImage(image) {
    // Redimensionar a imagem para 64x64 pixels
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([64, 64]).toFloat();
    // Normalizar a imagem
    tensor = tensor.div(tf.scalar(255.0));
    // Adicionar uma dimensão extra para o batch
    tensor = tensor.expandDims();
    return tensor;
}

// Executar inferência
async function runInference(model, image) {
    const inputTensor = preprocessImage(image);
    const prediction = model.predict(inputTensor);
    prediction.print();
    return prediction;
}

// Função principal
async function main() {
    const model = await loadModel();
    const image = document.getElementById('input-image'); // Supondo que a imagem de entrada tenha o ID 'input-image'
    await runInference(model, image);
}

main();
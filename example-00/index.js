import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // InputShape: Primeira camada da rede
    // Entrada com 7 posições (idade normalizada + 3 cores + 3 locais)

    // Units: 80 neuronios = essa quantidade pois a base de treino é pequena
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e maior é o processamento que ela vai usar

    // activation:ReLu: age como um filtro, se chegar na rede processada e o valor for positivo
    // considera o valor, se for 0 ou negativo, descarta
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));



    // Saida de dados: 
    // Units: 3 neurônios, pois são 3 categorias
    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))


    // Compilando o modelo - define qual sera o processo que ele vai usar para aprender
    // optimizer:adam(adaptative moment estimation) - é um treinador pessoal moderno, ajusta os pessos de forma eficiente e inteligente
    // aprende com histórico de erros e acertos
    // loss: categoricalCrossentropy - ele compara o que o modelo acha (scores de cada categoria) com a resposta certa
    // metrics: accuracy - mede quão certo/errado está em relação aos dados de treino
    // categoricalCrossentropy - classificação de imagens, recomendação e categorização de usuário - qualquer coisa que a resposta certa é sempre uma opção
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })


    // Treinamento do modelo
    // shuffle: vai embaralhar os dados a cada iteração
    // epochs: quantas vezes ele vai passar pelos nossos dados de treinamento
    await model.fit(inputXs, outputYs, { verbose: 0, epochs: 100, shuffle: true, callbacks: { onEpochEnd: (epoch, log) => console.log(`Epoch: ${epoch}: loss = ${log.loss}`) } })

    return model;
}

async function predict(model, normalizedPersonTensor) {
    // transformar os dados em tipos tensor
    const tfInput = tf.tensor2d(normalizedPersonTensor)

    const pred = model.predict(tfInput)
    const predArray = await pred.array();
    
    return predArray[0].map((prob, index) => ({ prob, index}))
}



// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0   , 0, 1, 0, 0, 1, 0],    // Ana
//     [1   , 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// normalizedPeopleTensor corresponde ao dataset de entrada do modelo.
const normalizedPeopleTensor = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(normalizedPeopleTensor) // Dados originais para treinamento do modelo
const outputYs = tf.tensor2d(tensorLabels) // categorias aplicadas aos dados originais, para o modelo saber como categorizar

const model = await trainModel(inputXs, outputYs)

// baseado no carlos
const person = { nome: 'Zé', idade: 28, cor: 'verde', localizacao: 'Curitiba'}
// normalizando
// exmplo: min_age = 25, max_age = 40, então (28 - 25) / (40 - 25) = 0.2
const normalizedPersonTensor = [
    [
        0.2, // idade normalizada
        0, // cor azul,
        0, // cor vermelho,
        1, // cor verde,
        0, // são paulo
        0, // rio
        1, // curitiba
    ]
]


const predictions = await predict(model, normalizedPersonTensor)
const results = predictions.sort((a,b) => b.prob - a.prob).map((p) => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`).join('\n')

console.log(results)
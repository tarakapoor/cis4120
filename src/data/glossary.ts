export interface TermDefinition {
  term: string;
  definition: string;
  example?: string;
  relatedTerms?: string[];
}

export const glossary: Record<string, TermDefinition> = {
  activation: {
    term: "activation",
    definition: "The output value of a neuron (node) after processing its input. Think of it as the neuron's 'answer' or 'response' to the information it receives. Higher values typically indicate stronger responses.",
    example: "If a neuron receives input about detecting an edge in an image, its activation might be 0.8 (very confident) or 0.2 (not confident).",
    relatedTerms: ["neuron", "layer", "output"]
  },
  neuron: {
    term: "neuron",
    definition: "A basic processing unit in a neural network (shown as a circle in the visualization). Each neuron takes inputs, processes them, and produces an output called an activation. Neurons are connected together to form the network.",
    example: "Similar to how brain cells work together, neurons in a neural network work together to solve problems.",
    relatedTerms: ["activation", "layer", "network"]
  },
  layer: {
    term: "layer",
    definition: "A group of neurons that process information at the same stage. Layers are organized from input (receiving data) to output (producing results), with hidden layers in between that transform the information.",
    example: "A network might have an input layer (receives images), hidden layers (processes features), and an output layer (produces predictions).",
    relatedTerms: ["neuron", "input", "output", "hidden layer"]
  },
  encoder: {
    term: "encoder",
    definition: "A part of a neural network that compresses or transforms input data into a more compact representation. It extracts important features from the raw input and represents them in a way that's useful for making decisions.",
    example: "An encoder might take a full image and compress it into a smaller set of numbers that represent the key features of that image.",
    relatedTerms: ["decoder", "layer", "feature"]
  },
  decoder: {
    term: "decoder",
    definition: "The counterpart to an encoder. It takes compressed or encoded information and expands it back into a more detailed format, often to reconstruct data or generate outputs.",
    example: "A decoder might take compressed image features and reconstruct them into a full image.",
    relatedTerms: ["encoder", "output", "reconstruction"]
  },
  policy: {
    term: "policy",
    definition: "In robotics and reinforcement learning, a policy is the strategy or set of rules that determines what action to take in each situation. It's like the robot's decision-making brain that tells it what to do based on what it observes.",
    example: "A robot's policy might say: 'If I see an obstacle in front, turn left. If the path is clear, move forward.'",
    relatedTerms: ["reinforcement learning", "action", "decision"]
  },
  network: {
    term: "neural network",
    definition: "A computing system inspired by biological neural networks. It consists of interconnected neurons organized in layers that work together to process information, recognize patterns, and make predictions.",
    example: "A neural network can learn to recognize cats in photos by seeing many examples and adjusting the connections between neurons.",
    relatedTerms: ["neuron", "layer", "machine learning"]
  },
  edge: {
    term: "edge",
    definition: "A connection (line) between two neurons in the network. Each edge has a weight that determines how strongly the signal from one neuron influences the next. These connections are what allow information to flow through the network.",
    example: "A thick, brightly colored edge indicates a strong connection where one neuron strongly influences another.",
    relatedTerms: ["neuron", "connection", "weight"]
  },
  connection: {
    term: "connection",
    definition: "Same as an edge - a link between neurons that allows information to flow from one to another. The strength of the connection (weight) determines how much influence one neuron has on another.",
    example: "When you click a neuron, you can see all its connections highlighted, showing how it communicates with other parts of the network.",
    relatedTerms: ["edge", "neuron", "weight"]
  },
  weight: {
    term: "weight",
    definition: "A numerical value that determines the strength of a connection between two neurons. During training, these weights are adjusted to help the network learn. Higher weights mean stronger influence.",
    example: "A weight of 0.9 means a very strong connection, while 0.1 means a weak connection.",
    relatedTerms: ["edge", "connection", "training"]
  },
  input: {
    term: "input layer",
    definition: "The first layer of the network that receives the raw data (like images, text, or sensor readings). This is where information enters the network.",
    example: "For an image recognition network, the input layer receives the pixel values of an image.",
    relatedTerms: ["layer", "output", "hidden layer"]
  },
  output: {
    term: "output layer",
    definition: "The final layer of the network that produces the final result or prediction. This is where the network's decision or answer comes out.",
    example: "For a network that classifies images, the output layer might produce probabilities like 'cat: 0.95, dog: 0.03, bird: 0.02'.",
    relatedTerms: ["layer", "input", "prediction"]
  },
  "hidden layer": {
    term: "hidden layer",
    definition: "Layers between the input and output layers that process and transform the information. They're called 'hidden' because their inner workings are complex and not directly visible in the input or output.",
    example: "Hidden layers might detect edges in early layers, then shapes in middle layers, and finally complex patterns in later layers.",
    relatedTerms: ["layer", "input", "output"]
  },
  feature: {
    term: "feature",
    definition: "A distinctive characteristic or pattern that the network learns to recognize. Features can be simple (like edges) or complex (like faces or objects).",
    example: "In image recognition, features might include edges, textures, shapes, or entire objects.",
    relatedTerms: ["layer", "pattern", "recognition"]
  },
  "reinforcement learning": {
    term: "reinforcement learning",
    definition: "A type of machine learning where an agent (like a robot) learns by interacting with its environment. It receives rewards for good actions and learns to maximize these rewards over time.",
    example: "A robot learns to walk by trying different movements and getting rewarded when it moves forward successfully.",
    relatedTerms: ["policy", "action", "reward"]
  },
  training: {
    term: "training",
    definition: "The process of teaching a neural network by showing it many examples and adjusting the weights of connections to improve its performance. This is how the network learns.",
    example: "Training a network to recognize cats involves showing it thousands of cat images and adjusting the network until it can correctly identify cats.",
    relatedTerms: ["weight", "learning", "network"]
  }
};

export function getDefinition(term: string): TermDefinition | undefined {
  const normalizedTerm = term.toLowerCase().trim();
  return glossary[normalizedTerm];
}

export function getAllTerms(): string[] {
  return Object.keys(glossary);
}


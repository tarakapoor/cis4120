/**
 * Utility functions for managing neural network models and weights
 */

export interface WeightData {
  sourceId: string;
  targetId: string;
  weight: number;
}

export interface ModelWithWeights {
  layers: Array<{ size: number }>;
  weights?: WeightData[];
}

/**
 * Initialize weights for a model if they don't exist
 * Generates random weights between -1 and 1 for all connections
 */
export function initializeWeights(model: any): ModelWithWeights {
  if (!model || !model.layers) return model;

  // If weights already exist, return as-is
  if (model.weights && Array.isArray(model.weights)) {
    return model;
  }

  const layers = model.layers;
  const weights: WeightData[] = [];

  // Generate weights for connections between layers
  for (let layerIdx = 0; layerIdx < layers.length - 1; layerIdx++) {
    const currentLayerSize = layers[layerIdx].size;
    const nextLayerSize = layers[layerIdx + 1].size;

    for (let i = 0; i < currentLayerSize; i++) {
      for (let j = 0; j < nextLayerSize; j++) {
        const sourceId = `L${layerIdx}-N${i}`;
        const targetId = `L${layerIdx + 1}-N${j}`;
        // Initialize with random weights between -1 and 1
        const weight = (Math.random() * 2 - 1) * 0.5; // Scale to -0.5 to 0.5 for better visualization
        weights.push({ sourceId, targetId, weight });
      }
    }
  }

  return {
    ...model,
    weights
  };
}

/**
 * Get weight between two nodes
 */
export function getWeight(weights: WeightData[] | undefined, sourceId: string, targetId: string): number {
  if (!weights) return 0;
  const weightData = weights.find(w => w.sourceId === sourceId && w.targetId === targetId);
  return weightData ? weightData.weight : 0;
}

/**
 * Update weight between two nodes
 */
export function updateWeight(
  weights: WeightData[] | undefined,
  sourceId: string,
  targetId: string,
  newWeight: number
): WeightData[] {
  if (!weights) return [];
  
  const updated = weights.map(w => 
    w.sourceId === sourceId && w.targetId === targetId
      ? { ...w, weight: newWeight }
      : w
  );
  
  // If weight didn't exist, add it
  const exists = weights.some(w => w.sourceId === sourceId && w.targetId === targetId);
  if (!exists) {
    updated.push({ sourceId, targetId, weight: newWeight });
  }
  
  return updated;
}

/**
 * Get all outgoing weights from a node
 */
export function getOutgoingWeights(weights: WeightData[] | undefined, sourceId: string): WeightData[] {
  if (!weights) return [];
  return weights.filter(w => w.sourceId === sourceId);
}

/**
 * Normalize weight to a visual representation (0-1 scale for opacity/thickness)
 */
export function normalizeWeight(weight: number): number {
  // Map weight from range [-1, 1] to [0, 1] for visualization
  return (weight + 1) / 2;
}

/**
 * Get color for weight visualization
 * - Positive weights: shades of blue (brighter/darker blue for larger magnitudes)
 * - Negative weights: shades of grey (darker grey for larger absolute values)
 * - Zero weights: neutral grey
 */
export function getWeightColor(weight: number, opacity: number = 1): string {
  const clampedOpacity = Math.max(0, Math.min(1, opacity));
  const absWeight = Math.min(Math.abs(weight), 1);
  
  if (weight > 0) {
    // Positive weights: shades of blue
    // Light blue (#93c5fd) for small weights → Dark blue (#1e40af) for large weights
    const intensity = absWeight; // 0 to 1
    
    // Interpolate between light blue and dark blue
    const r = Math.round(147 - (intensity * 113)); // 147 -> 34
    const g = Math.round(197 - (intensity * 157)); // 197 -> 40
    const b = Math.round(253 - (intensity * 163)); // 253 -> 90
    
    const alpha = 0.5 + (intensity * 0.5); // Range from 0.5 to 1.0
    return `rgba(${r}, ${g}, ${b}, ${alpha * clampedOpacity})`;
  } else if (weight < 0) {
    // Negative weights: shades of grey
    // Light grey for small absolute values → Dark grey/black for large absolute values
    const intensity = absWeight;
    
    // Interpolate between light grey (#cbd5e1) and dark grey (#475569)
    const r = Math.round(203 - (intensity * 128)); // 203 -> 71
    const g = Math.round(213 - (intensity * 128)); // 213 -> 85
    const b = Math.round(225 - (intensity * 112)); // 225 -> 105
    
    const alpha = 0.4 + (intensity * 0.5); // Range from 0.4 to 0.9
    return `rgba(${r}, ${g}, ${b}, ${alpha * clampedOpacity})`;
  } else {
    // Zero weight: neutral grey
    return `rgba(148, 163, 184, ${0.5 * clampedOpacity})`; // #94a3b8 at 50% opacity
  }
}

/**
 * Get stroke width for weight visualization
 */
export function getWeightStrokeWidth(weight: number, baseWidth: number = 1.2): number {
  const absWeight = Math.abs(weight);
  // Scale from 1.2 to 5 based on weight magnitude
  return baseWidth + (absWeight * 3.8);
}


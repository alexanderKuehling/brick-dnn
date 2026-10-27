import { createContext, useContext } from 'react';
import type { Analysis, LayerSpec } from '@/api/types';
import type { LayerNodeData } from './graph';

export interface BuilderContextValue {
  specs: Record<string, LayerSpec>;
  analysis: Analysis | null;
  errorsByNode: Record<string, string[]>;
  warningsByNode: Record<string, string[]>;
  updateNode: (id: string, patch: Partial<LayerNodeData>) => void;
  deleteNode: (id: string) => void;
  duplicateNode: (id: string) => void;
  focusNode: (id: string) => void;
}

export const BuilderContext = createContext<BuilderContextValue | null>(null);

export function useBuilder(): BuilderContextValue {
  const context = useContext(BuilderContext);
  if (!context) {
    throw new Error('useBuilder must be used inside the builder');
  }
  return context;
}

export function groupIssues(issues: { node: string | null; message: string }[] = []) {
  const grouped: Record<string, string[]> = {};
  for (const issue of issues) {
    if (issue.node) {
      (grouped[issue.node] ??= []).push(issue.message);
    }
  }
  return grouped;
}

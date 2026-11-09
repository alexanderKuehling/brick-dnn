import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { loadDraft, saveDraft, type Draft } from './draft';

/**
 * Opens a network in the builder (from other pages). Asks before replacing a different network
 * that is open there, since its unsaved changes would be lost.
 */
export function useOpenInBuilder(): (draft: Draft) => void {
  const navigate = useNavigate();
  return useCallback(
    (draft: Draft) => {
      const current = loadDraft();
      const replace = () => {
        saveDraft(draft);
        navigate('/');
      };
      const sameProject = draft.projectId !== null && current?.projectId === draft.projectId;
      if (!current || current.doc.nodes.length === 0 || current.saved || sameProject) {
        replace();
        return;
      }
      modals.openConfirmModal({
        title: 'Replace the network in the builder?',
        children: (
          <Text size="sm">
            “{current.name}” is currently open in the builder. Unsaved changes to it will be lost.
          </Text>
        ),
        labels: { confirm: 'Replace', cancel: 'Cancel' },
        onConfirm: replace,
      });
    },
    [navigate]
  );
}

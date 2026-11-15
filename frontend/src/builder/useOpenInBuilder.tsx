import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { i18n } from '@/i18n';
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
        title: i18n.t('builder.page.replaceInBuilder'),
        children: (
          <Text size="sm">
            {i18n.t('builder.page.replaceInBuilderText', { name: current.name })}
          </Text>
        ),
        labels: { confirm: i18n.t('builder.page.replace'), cancel: i18n.t('common.cancel') },
        onConfirm: replace,
      });
    },
    [navigate]
  );
}

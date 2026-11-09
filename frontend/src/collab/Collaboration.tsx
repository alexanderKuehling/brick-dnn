import { useCallback, useEffect } from 'react';
import { IconHistory, IconMessageCircle } from '@tabler/icons-react';
import { Panel, useNodes } from '@xyflow/react';
import { Badge, Box, Button, CloseButton, Drawer, Tabs, useMantineTheme } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import type { Project, ProjectRole } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { useBuilder } from '@/builder/BuilderContext';
import type { LayerFlowNode } from '@/builder/graph';
import { CommentsPanel } from '@/comments/CommentsPanel';
import { HEADER_HEIGHT } from '@/components/AppHeader';
import { DEMO } from '@/demo';
import { HistoryPanel } from '@/history/HistoryPanel';
import { useCollab, useCollabStore } from './context';
import type { SubscribeToEvents } from './events';
import type { CollabPanel, CollabStore } from './store';
import { usePolling, useProjectEvents } from './useProjectEvents';
import classes from './Collab.module.css';

/** Without a live connection, the open comments panel loads the comments this often. */
export const COMMENTS_POLL_MS = 15_000;

export interface CollaborationProps {
  /** The saved project on the canvas; null: none (a new network). */
  projectId: number | null;
  /** The user's role in it (null: not known yet, treated as view only). */
  role: ProjectRole | null;
  /**
   * The live messages of the project (`useLiveBuilder(...).subscribe`): comments and the
   * history follow them. Without it, or while `live` is false, the open panel polls.
   */
  subscribe?: SubscribeToEvents | null;
  /** The live connection is up. */
  live?: boolean;
  /** The canvas has changes that are not saved (named versions and restores leave them out). */
  unsaved?: boolean;
  /**
   * A version was restored: the project as it is now. Show it on the canvas, unless the live
   * connection does (it gets a `replaced` message as well).
   */
  onRestored: (project: Project) => void;
}

/**
 * History and comments of a saved network in the builder: two buttons in the top right
 * corner of the canvas (render it inside `<ReactFlow>`) and the side panel they open.
 * Needs a `CollabProvider` above the canvas, which also feeds the comment markers on the
 * layers (`CommentMarker`) and the comments in the inspector (`LayerComments`). Nothing is
 * shown for networks that are not saved, without login and in the online demo.
 */
export function Collaboration({
  projectId,
  role,
  subscribe,
  live = false,
  unsaved = false,
  onRestored,
}: CollaborationProps) {
  const store = useCollabStore();
  const { user } = useAuth();
  const theme = useMantineTheme();
  const wideScreen = useMediaQuery(`(min-width: ${theme.breakpoints.lg})`);
  const phone = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
  const active = !DEMO && user !== null && projectId !== null && store !== null;
  const openId = active ? projectId : null;
  const userId = user?.id ?? null;

  useEffect(() => {
    store?.configure({ projectId: openId, userId, role });
  }, [store, openId, userId, role]);

  // Live: comments of others arrive as messages; a reconnect may have missed some.
  useProjectEvents(active ? subscribe : null, {
    onComment: (event) => store?.applyEvent(event),
    onConnected: () => void store?.load(),
  });
  const panel = useCollab((state) => state.panel, null);
  const openCount = useCollab((state) => state.openCount, 0);
  // Not live: the open comments are loaded again now and then.
  usePolling(() => void store?.load(), COMMENTS_POLL_MS, active && !live && panel === 'comments');
  // Opening the comments shows the latest state.
  useEffect(() => {
    if (panel === 'comments') {
      void store?.load();
    }
  }, [panel, store]);

  if (!active || openId === null || !store) {
    return null;
  }

  return (
    <>
      <Panel position="top-right">
        <CollabButtons store={store} panel={panel} openCount={openCount} />
      </Panel>
      <Drawer
        opened={panel !== null}
        onClose={() => store.closePanel()}
        position="right"
        // On wide screens it covers the inspector column, so the canvas stays as it is.
        size={wideScreen ? 330 : phone ? '100%' : 380}
        padding={0}
        withCloseButton={false}
        // A side panel: the canvas stays usable next to it (it has a close button).
        withOverlay={false}
        trapFocus={false}
        lockScroll={false}
        closeOnEscape={false}
        shadow="md"
        classNames={{ inner: classes.drawerInner }}
        style={{ '--collab-top': `${HEADER_HEIGHT}px` } as React.CSSProperties}
      >
        <Tabs
          value={panel}
          onChange={(value) => value && store.openPanel(value as CollabPanel)}
          keepMounted={false}
        >
          <div className={classes.panelHeader}>
            <Tabs.List grow className={classes.tabs}>
              <Tabs.Tab value="history" leftSection={<IconHistory size={16} />}>
                History
              </Tabs.Tab>
              <Tabs.Tab
                value="comments"
                leftSection={<IconMessageCircle size={16} />}
                rightSection={<OpenBadge count={openCount} />}
              >
                Comments
              </Tabs.Tab>
            </Tabs.List>
            <CloseButton
              className={classes.close}
              onClick={() => store.closePanel()}
              aria-label="Close the panel"
            />
          </div>
          <Tabs.Panel value="history" p="md">
            <HistoryPanel
              projectId={openId}
              role={role}
              subscribe={subscribe}
              live={live}
              unsaved={unsaved}
              onRestored={onRestored}
            />
          </Tabs.Panel>
          <Tabs.Panel value="comments" p="md">
            <BuilderComments store={store} />
          </Tabs.Panel>
        </Tabs>
      </Drawer>
    </>
  );
}

function OpenBadge({ count }: { count: number }) {
  if (count === 0) {
    return null;
  }
  return (
    <Badge size="xs" circle color="yellow" c="black" variant="filled" aria-label={`${count} open`}>
      {count}
    </Badge>
  );
}

/** The comments panel with the layers of the canvas (only rendered while it is open). */
function BuilderComments({ store }: { store: CollabStore }) {
  const nodes = useNodes<LayerFlowNode>();
  const { specs, focusNode } = useBuilder();
  const theme = useMantineTheme();
  const wideScreen = useMediaQuery(`(min-width: ${theme.breakpoints.lg})`);

  const layerName = useCallback(
    (id: string) => {
      const node = nodes.find((n) => n.id === id);
      return node ? node.data.name || specs[node.data.type]?.label || node.data.type : null;
    },
    [nodes, specs]
  );
  const focusLayer = useCallback(
    (id: string) => {
      if (!wideScreen) {
        store.closePanel(); // The panel covers part of the canvas.
      }
      focusNode(id);
    },
    [wideScreen, store, focusNode]
  );

  return <CommentsPanel store={store} layerName={layerName} onFocusLayer={focusLayer} />;
}

interface CollabButtonsProps {
  store: CollabStore;
  panel: CollabPanel | null;
  openCount: number;
}

/** "History" and "Comments" (with the number of open threads) over the canvas. */
function CollabButtons({ store, panel, openCount }: CollabButtonsProps) {
  const toggle = (target: CollabPanel) =>
    panel === target ? store.closePanel() : store.openPanel(target);

  return (
    // Phones show the icons only (and the number of open threads).
    <div className={classes.buttons}>
      <Button
        size="xs"
        variant={panel === 'history' ? 'light' : 'default'}
        onClick={() => toggle('history')}
        aria-label="History"
        aria-pressed={panel === 'history'}
        title="Earlier versions of this network"
      >
        <IconHistory size={16} />
        <Box component="span" visibleFrom="xs" className={classes.buttonLabel}>
          History
        </Box>
      </Button>
      <Button
        size="xs"
        variant={panel === 'comments' ? 'light' : 'default'}
        onClick={() => toggle('comments')}
        aria-label={openCount === 0 ? 'Comments' : `Comments (${openCount} open)`}
        aria-pressed={panel === 'comments'}
        title="Comments on this network and its layers"
      >
        <IconMessageCircle size={16} />
        <Box component="span" visibleFrom="xs" className={classes.buttonLabel}>
          Comments
        </Box>
        {openCount > 0 && (
          <Box component="span" className={classes.buttonLabel}>
            <OpenBadge count={openCount} />
          </Box>
        )}
      </Button>
    </div>
  );
}

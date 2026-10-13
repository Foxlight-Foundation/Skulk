import { useRef, useState } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';
import {
  useGetBurstBudgetQuery,
  useGetBurstRentalsQuery,
  useReleaseBurstMutation,
  type BurstBudget,
  type BurstOperation,
  type BurstProviderAddress,
} from '../../store/endpoints/burst';
import { useGetPluginNodesQuery } from '../../store/endpoints/plugins';
import { Button } from '../common/Button';
import { burstPhaseLabels, formatCents, positiveUsd, spentOfLimit } from '../models/burstPresentation';
import { ProposalControls } from '../pages/NodeProposalsPanel';
import { amountFits, limitField, takesNoValues, type LimitField } from './limitField';
import { useSetupActionRun, type SetupActionRun } from './useSetupActionRun';

/** How often the view reads spending and rentals again while open. */
const VIEW_POLL_MS = 15_000;

const View = styled.div`
  display: flex;
  flex-direction: column;
  gap: 18px;
  overflow-wrap: anywhere;
`;

const Section = styled.section`
  h4 { margin: 0 0 8px; font-size: inherit; }
  p { margin: 6px 0; }
`;

const Facts = styled.dl`
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 14px;
  margin: 0;
  dt { color: ${({ theme }) => theme.colors.metadataText}; }
  dd { margin: 0; }
`;

const Actions = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
`;

const Card = styled.div`
  padding: 10px 12px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 8px;
  background: ${({ theme }) => theme.colors.surface};
`;

const Muted = styled.p`
  color: ${({ theme }) => theme.colors.metadataText};
`;

const AmountInput = styled.input`
  width: 9em;
  margin-left: 4px;
`;

const when = (seconds: number) => new Date(seconds * 1000).toLocaleString();
const whenDay = (seconds: number) => new Date(seconds * 1000).toLocaleDateString();

/** The limit as an editable amount in dollars and cents, from an exact dollar string. */
function editableAmount(value: string | null): string {
  if (value === null) return '';
  return formatCents(value, 'down').slice(1).replace(/,/g, '');
}

/** A typed amount in dollars and cents, or null when it is not a usable limit for this field. */
function parsedAmount(text: string, field: LimitField | null): number | null {
  if (field === null || !/^\d+(\.\d{1,2})?$/.test(text.trim())) return null;
  const amount = Number(text);
  return amountFits(field, amount) ? amount : null;
}

/** Dollars for a bound in the range message. */
const boundText = (value: number) => `$${value.toFixed(2)}`;

/** Props for the Rentals view of one burst provider node. */
export type BurstRentalsViewProps = BurstProviderAddress;

/**
 * A burst provider's rented machines, its spending, and the rentals waiting
 * for approval, as the capability panel's Rentals tab.
 *
 * Spending shows what finished rentals used and what running machines hold,
 * in dollars and cents. Where the provider allows it from this computer, the
 * limit can be changed or the count started over through the provider's own
 * setup actions. Running machines can be released here, and rentals the
 * steward proposed can be reviewed and approved.
 */
export function BurstRentalsView({ pluginId, nodeId }: BurstRentalsViewProps) {
  const { t } = useSkulkTranslation();
  const address = { pluginId, nodeId };
  const budgetQuery = useGetBurstBudgetQuery(address, {
    pollingInterval: VIEW_POLL_MS,
    skipPollingIfUnfocused: true,
    refetchOnMountOrArgChange: true,
  });
  const rentalsQuery = useGetBurstRentalsQuery(undefined, {
    pollingInterval: VIEW_POLL_MS,
    skipPollingIfUnfocused: true,
    refetchOnMountOrArgChange: true,
  });
  const pluginNodes = useGetPluginNodesQuery();
  const node = pluginNodes.currentData
    ?.find((plugin) => plugin.pluginId === pluginId)
    ?.nodes.find((candidate) => candidate.nodeId === nodeId);
  const rentals = (rentalsQuery.currentData?.rentals ?? []).filter(
    (rental) => rental.pluginId === pluginId && rental.nodeId === nodeId && rental.phase !== 'released',
  );

  return (
    <View>
      <SpendingSection address={address} budget={budgetQuery.currentData} unavailable={budgetQuery.isError && !budgetQuery.currentData} />
      <Section aria-label={t('burst.rentals.runningTitle', 'Running now')}>
        <h4>{t('burst.rentals.runningTitle', 'Running now')}</h4>
        {rentalsQuery.isError && !rentalsQuery.currentData ? (
          <Muted>{t('burst.rentals.rentalsUnavailable', 'Rented machines cannot be listed right now. This is checked again shortly.')}</Muted>
        ) : rentals.length === 0 ? (
          <Muted>{t('burst.rentals.noneRunning', 'No rented machines are running.')}</Muted>
        ) : (
          rentals.map((rental) => <RentalCard key={rental.operationId} rental={rental} />)
        )}
      </Section>
      {node?.proposalsAvailable ? (
        <Section aria-label={t('burst.rentals.waitingTitle', 'Waiting for your approval')}>
          <h4>{t('burst.rentals.waitingTitle', 'Waiting for your approval')}</h4>
          <p>
            {t(
              'burst.rentals.waitingIntro',
              'Rentals the steward proposed wait here for you. Nothing is rented until you approve one.',
            )}
          </p>
          <ProposalControls actionsAvailable={node.proposalActionsAvailable === true} nodeId={nodeId} pluginId={pluginId} />
        </Section>
      ) : null}
    </View>
  );
}

/** Spending: what is used of the limit, and the provider's own ways to change it. */
function SpendingSection({
  address,
  budget,
  unavailable,
}: {
  address: BurstProviderAddress;
  budget: BurstBudget | undefined;
  unavailable: boolean;
}) {
  const { t } = useSkulkTranslation();
  const [mode, setMode] = useState<'view' | 'limit' | 'startOver'>('view');
  const limitRun = useSetupActionRun(address, mode === 'limit');
  const startOverRun = useSetupActionRun(address, mode === 'startOver');
  const spent = spentOfLimit(budget, t);
  const title = <h4>{t('burst.rentals.spendingTitle', 'Spending')}</h4>;

  if (!budget) {
    return (
      <Section aria-label={t('burst.rentals.spendingTitle', 'Spending')}>
        {title}
        <Muted>
          {unavailable
            ? t('burst.rentals.budgetUnavailable', 'Your spending limit cannot be read right now. This is checked again shortly.')
            : t('burst.rentals.budgetLoading', 'Reading your spending limit…')}
        </Muted>
      </Section>
    );
  }
  if (budget.state !== 'available' || !spent) {
    return (
      <Section aria-label={t('burst.rentals.spendingTitle', 'Spending')}>
        {title}
        <p>{budget.message ?? t('burst.rentals.budgetUnavailable', 'Your spending limit cannot be read right now. This is checked again shortly.')}</p>
      </Section>
    );
  }

  const close = () => {
    limitRun.reset();
    startOverRun.reset();
    setMode('view');
  };

  return (
    <Section aria-label={t('burst.rentals.spendingTitle', 'Spending')}>
      {title}
      <Facts>
        <dt>{t('burst.rentals.spentLabel', 'Spent')}</dt>
        <dd>{spent}</dd>
        {budget.leftUsd !== null ? (
          <>
            <dt>{t('burst.rentals.leftLabel', 'Left')}</dt>
            <dd>{formatCents(budget.leftUsd, 'down')}</dd>
          </>
        ) : null}
        {positiveUsd(budget.heldUsd) && budget.heldUsd !== null ? (
          <>
            <dt>{t('burst.rentals.heldLabel', 'Held')}</dt>
            <dd>{t('burst.rentals.heldNow', '{held} held for machines running now', { held: formatCents(budget.heldUsd) })}</dd>
          </>
        ) : null}
      </Facts>
      {budget.countingSince !== null ? (
        <Muted>{t('burst.rentals.countingSince', 'Counting since {date}', { date: whenDay(budget.countingSince) })}</Muted>
      ) : null}
      {budget.message ? <p>{budget.message}</p> : null}
      {mode === 'view' ? (
        <Actions>
          {budget.changeLimitAction ? (
            <Button onClick={() => setMode('limit')} type="button">
              {t('burst.rentals.changeLimit', 'Change limit')}
            </Button>
          ) : null}
          {budget.startOverAction ? (
            <Button onClick={() => setMode('startOver')} type="button">
              {t('burst.rentals.startOver', 'Start over')}
            </Button>
          ) : null}
        </Actions>
      ) : null}
      {mode === 'limit' && budget.changeLimitAction ? (
        <LimitForm actionId={budget.changeLimitAction} current={budget.limitUsd} onClose={close} run={limitRun} />
      ) : null}
      {mode === 'startOver' && budget.startOverAction ? (
        <StartOverConfirm actionId={budget.startOverAction} onClose={close} run={startOverRun} />
      ) : null}
    </Section>
  );
}

/** The outcome of a run, in plain words, or nothing while idle. */
function RunOutcome({ run, done }: { run: SetupActionRun; done: string }) {
  const { t } = useSkulkTranslation();
  if (run.phase === 'running') return <p role="status">{t('burst.rentals.saving', 'Saving…')}</p>;
  if (run.phase === 'complete') return <p role="status">{done}</p>;
  if (run.phase === 'failed') return <p role="alert">{run.detail ?? t('burst.rentals.notChanged', 'Not changed. Try again.')}</p>;
  return null;
}

/** The range a limit must fall in, in plain words, from the provider's own form. */
function rangeMessage(field: LimitField, t: SkulkTranslate): string {
  const low =
    field.minimum !== null && field.minimum > 0
      ? t('burst.rentals.limitAtLeast', 'at least {amount}', { amount: boundText(field.minimum) })
      : t('burst.rentals.limitMoreThan', 'more than {amount}', { amount: boundText(field.exclusiveMinimum ?? 0) });
  return field.maximum !== null
    ? t('burst.rentals.limitRangeBounded', 'Enter an amount of {low} and at most {high}.', {
        low,
        high: boundText(field.maximum),
      })
    : t('burst.rentals.limitRangeOpen', 'Enter an amount of {low}.', { low });
}

/**
 * One dollar field, prefilled with today's limit. The field's name and bounds
 * come from the provider's own form, so the dashboard assumes neither.
 */
function LimitForm({
  actionId,
  current,
  run,
  onClose,
}: {
  actionId: string;
  current: string | null;
  run: SetupActionRun;
  onClose: () => void;
}) {
  const { t } = useSkulkTranslation();
  const [text, setText] = useState(() => editableAmount(current));
  const field = run.ready ? limitField(run.action(actionId)?.parametersSchema) : null;
  const amount = parsedAmount(text, field);
  const submitting = run.phase === 'running';
  const finished = run.phase === 'complete';
  if (run.ready && field === null && run.phase === 'idle') {
    return (
      <Card>
        <p role="alert">
          {t(
            'burst.rentals.limitFormUnavailable',
            "This provider's limit can't be changed here. Change it in the provider's own setup.",
          )}
        </p>
        <Actions>
          <Button onClick={onClose} type="button">
            {t('burst.rentals.cancel', 'Cancel')}
          </Button>
        </Actions>
      </Card>
    );
  }
  return (
    <Card>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (amount !== null && field !== null && run.ready && !submitting) {
            void run.start(actionId, { [field.name]: amount });
          }
        }}
      >
        <label>
          {t('burst.rentals.newLimit', 'Spend no more than')} $
          <AmountInput
            aria-label={t('burst.rentals.newLimitAria', 'New spending limit in dollars')}
            disabled={submitting || finished}
            inputMode="decimal"
            onChange={(event) => setText(event.target.value)}
            value={text}
          />
        </label>
        <Muted>
          {t(
            'burst.rentals.newLimitHelp',
            'The most all your rentals together may cost, counting what they have already used.',
          )}
        </Muted>
        {field !== null && text.trim() !== '' && amount === null ? <p role="alert">{rangeMessage(field, t)}</p> : null}
        <RunOutcome
          done={t('burst.rentals.limitChanged', 'Your limit is now {limit}.', {
            limit: amount !== null ? `$${amount.toFixed(2)}` : '',
          })}
          run={run}
        />
        <Actions>
          {!finished ? (
            <Button disabled={amount === null || field === null || submitting} type="submit">
              {t('burst.rentals.save', 'Save')}
            </Button>
          ) : null}
          <Button disabled={submitting} onClick={onClose} type="button">
            {finished ? t('burst.rentals.done', 'Done') : t('burst.rentals.cancel', 'Cancel')}
          </Button>
        </Actions>
      </form>
    </Card>
  );
}

/** A confirm step before the count starts over. */
function StartOverConfirm({ actionId, run, onClose }: { actionId: string; run: SetupActionRun; onClose: () => void }) {
  const { t } = useSkulkTranslation();
  const submitting = run.phase === 'running';
  const finished = run.phase === 'complete';
  // One button can only run an action whose form takes no values.
  if (run.ready && run.phase === 'idle' && !takesNoValues(run.action(actionId)?.parametersSchema)) {
    return (
      <Card>
        <p role="alert">
          {t(
            'burst.rentals.startOverUnavailable',
            "This provider can't start over from here. Use the provider's own setup.",
          )}
        </p>
        <Actions>
          <Button onClick={onClose} type="button">
            {t('burst.rentals.cancel', 'Cancel')}
          </Button>
        </Actions>
      </Card>
    );
  }
  return (
    <Card>
      <p>
        {t(
          'burst.rentals.startOverExplain',
          'Finished rentals stop counting, so your whole limit is available again. A machine running now keeps counting.',
        )}
      </p>
      <RunOutcome done={t('burst.rentals.startedOver', 'Started over. Finished rentals no longer count.')} run={run} />
      <Actions>
        {!finished ? (
          <Button disabled={!run.ready || submitting} onClick={() => void run.start(actionId, {})} type="button">
            {t('burst.rentals.startOverConfirm', 'Start over')}
          </Button>
        ) : null}
        <Button disabled={submitting} onClick={onClose} type="button">
          {finished ? t('burst.rentals.done', 'Done') : t('burst.rentals.cancel', 'Cancel')}
        </Button>
      </Actions>
    </Card>
  );
}

/** One rented machine: its model, where it stands, when it ends, and Release. */
function RentalCard({ rental }: { rental: BurstOperation }) {
  const { t } = useSkulkTranslation();
  const phaseLabels = burstPhaseLabels(t);
  const [release, releasing] = useReleaseBurstMutation();
  const [notice, setNotice] = useState('');
  const sending = useRef(false);
  const releasable = !['released', 'refused', 'releasing'].includes(rental.phase);
  const releaseRental = async () => {
    if (sending.current) return;
    sending.current = true;
    setNotice('');
    try {
      await release({ pluginId: rental.pluginId, nodeId: rental.nodeId, operationId: rental.operationId }).unwrap();
      setNotice(t('burst.rentals.releasing', 'Releasing the machine.'));
    } catch {
      setNotice(
        t(
          'burst.rentals.releaseFailed',
          'The release was not confirmed. Try again. The rental still ends on its own by its end time.',
        ),
      );
    } finally {
      sending.current = false;
    }
  };
  return (
    <Card aria-label={rental.modelId ?? t('burst.rentals.someModel', 'A model')}>
      <p>
        <strong>{rental.modelId ?? t('burst.rentals.someModel', 'A model')}</strong> · {phaseLabels[rental.phase]}
      </p>
      {rental.resourceExpiresAt ? (
        <Muted>{t('burst.rentals.endsBy', 'Ends by {time}', { time: when(rental.resourceExpiresAt) })}</Muted>
      ) : null}
      {notice ? <p role="status">{notice}</p> : null}
      {releasable ? (
        <Actions>
          <Button disabled={releasing.isLoading} onClick={() => void releaseRental()} type="button">
            {t('burst.rentals.release', 'Release')}
          </Button>
        </Actions>
      ) : null}
    </Card>
  );
}

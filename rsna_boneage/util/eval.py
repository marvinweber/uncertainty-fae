from datetime import datetime, timedelta


def print_progress(position: int, data_length: int, start: datetime):
    now = datetime.now()
    runtime_done = (now - start).seconds
    remaining_prop = data_length / position
    expected_runtime = runtime_done * remaining_prop
    remaining_seconds = expected_runtime - runtime_done
    expected_end_datetime = now + timedelta(seconds=remaining_seconds)

    percentage_done = round(position / data_length * 100, 1)
    minutes_done = round(runtime_done / 60, 2)
    minutes_remaining = round(remaining_seconds / 60, 2)
    expected_minutes = round(minutes_done + minutes_remaining, 2)

    print(f'Progress {position}/{data_length} -- {percentage_done} % -- '
          f'{minutes_done}/{expected_minutes} Minutes -- '
          f'Expected End: {expected_end_datetime.strftime("%y-%m-%d %H:%M")}; '
          f'in {minutes_remaining} Minutes.')

    return percentage_done, minutes_done, minutes_remaining, expected_end_datetime, expected_minutes

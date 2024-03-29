select * from (
    select distinct content, array_reverse(split(sample_path, '.'))[0] as ext, repo_name, sample_path as path
    from `bigquery-public-data.github_repos.sample_contents` as a
    inner join (
        select distinct repo_name, license
        from `bigquery-public-data.github_repos.licenses`
        where license in ('apache-2.0','bsd-2-clause', 'bsd-3-clause', 'mit')
    ) as b
    on a.sample_repo_name = b.repo_name
    where sample_path is not null and size < 100000 and size > 1024
    order by rand()
)
where ext in ('java', 'js', 'cs', 'h', 'py', 'php', 'rb', 'cpp', 'go', 'c', 'cc', 'hpp', 'scala', 'swift', 'sh', 'rst', 'scss', 'sql', 'coffee', 'groovy', 'st', 'hs', 'less', 'lua', 'hx', 'jsx', 'cshtml', 'mm', 'cmake')
limit 100000

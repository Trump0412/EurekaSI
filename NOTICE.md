# Ownership and upstream code

The public infrastructure is now named EurekaSI (https://github.com/Trump0412/EurekaSI). The earlier collective attribution in LICENSE is preserved; the rename does not change licensing scope or upstream ownership.

`spatial_intelligence/` is newly authored shared infrastructure, except `vendor/vsi/`, which preserves its upstream license and attribution. Internal reproduction bundles may contain `legacy_sources/` snapshots of the user's three repositories. Public source archives omit them by default and retain the commits and patches listed in `sources.lock.json` and `patches/`. Those snapshots are not maintained copies of the new framework.

Reference implementations GeoThinker, GeoSR, SpatialStack, EASI, VGGT, DA3, Pi3 and LIBERO are linked and fetched to the external workspace at pinned commits. Their code, datasets and weights retain their respective licenses. No third-party model weights or datasets are redistributed in this archive.

Newly authored shared code is provided under the scoped MIT license in LICENSE. This does not add permission to historical or third-party material with different or unspecified licensing. For public release, omit any historical snapshot whose redistribution rights have not been confirmed; the pinned source fetcher and patches preserve a reproduction path without vendoring it.
